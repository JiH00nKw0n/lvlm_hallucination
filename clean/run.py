"""Single entry point for all 4 experiment families.

Usage:
    python -m clean.run <config.yaml> [--stage all|extract|train|perm|eval|plot]

Pipeline dispatch is by `kind:` in the YAML. Each pipeline is idempotent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `clean.*` imports resolve no matter where
# the user runs this from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from clean.src.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to YAML config")
    p.add_argument("--stage", type=str, default="all",
                   choices=["all", "extract", "train", "perm", "eval", "plot", "density"])
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    logger = logging.getLogger("clean.run")
    logger.info("Loading config: %s", args.config)
    cfg = load_config(args.config)
    logger.info("kind=%s output_root=%s", cfg.kind, cfg.output.root)

    from clean.src.pipelines.cc3m_downstream import run as cc3m_run
    from clean.src.pipelines.multi_density import run as md_run
    from clean.src.pipelines.synthetic_sweep import run as ss_run

    pipelines = {
        "cc3m_downstream": cc3m_run,
        "multi_density": md_run,
        "synthetic_sweep": ss_run,
    }
    try:
        fn = pipelines[cfg.kind]
    except KeyError:
        raise SystemExit(f"Unknown kind {cfg.kind!r}; expected one of {list(pipelines)}")

    fn(cfg, stage=args.stage)


if __name__ == "__main__":
    main()
