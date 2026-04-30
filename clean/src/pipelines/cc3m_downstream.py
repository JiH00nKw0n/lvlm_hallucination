"""CC3M-trained downstream pipeline.

Steps:
  1. Extract paired (image, text) embeddings into `cfg.cache.cache_dir`.
  2. Train each method's SAE.
  3. Build ONE Hungarian perm from CC3M train (used by `Ours`).
  4. Run downstream evals (retrieval, zero-shot, steering, MS).
  5. Aggregate into `<output_root>/table.md`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from clean.src.alignment import build_perm, save_perm, load_perm
from clean.src.data.extract import extract_cache
from clean.src.training.trainer import train_method
from clean.src.utils.config import Config, MethodConfig

logger = logging.getLogger(__name__)


def _step(name: str, fn, *, output_marker: Path | None = None):
    if output_marker and output_marker.exists():
        logger.info("[skip] %s — exists: %s", name, output_marker)
        return
    logger.info("[run]  %s", name)
    fn()


def run(cfg: Config, stage: str = "all") -> None:
    assert cfg.kind == "cc3m_downstream", f"Wrong kind: {cfg.kind}"
    assert cfg.model is not None and cfg.cache is not None
    out_root = Path(cfg.output.root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Extract cache (idempotent inside).
    if stage in ("all", "extract"):
        extract_cache(model_cfg=cfg.model, cache_cfg=cfg.cache,
                      batch_size=64, num_workers=2, device=cfg.training.device)

    if stage == "extract":
        return

    hidden_size = cfg.model.hidden_size
    method_artifacts: dict[str, Path] = {}

    # 2) Train every method declared in cfg.methods.
    if stage in ("all", "train"):
        for method in cfg.methods:
            save_dir = out_root / method.name
            arts = train_method(
                method=method, training=cfg.training,
                cache_dir=cfg.cache.cache_dir, hidden_size=hidden_size,
                save_dir=save_dir,
            )
            method_artifacts[method.name] = arts.save_dir

    # 3) Build single Hungarian perm from CC3M train (only for `ours`).
    has_ours = any(m.name == "ours" for m in cfg.methods)
    perm_path = out_root / "ours" / "perm.npz"
    if has_ours and stage in ("all", "perm"):
        if not perm_path.exists():
            from clean.src.models import TwoSidedTopKSAE
            sep_dir = out_root / "separated" / "final"
            if not sep_dir.exists():
                logger.warning("[perm] separated ckpt missing at %s — skip", sep_dir)
            else:
                model = TwoSidedTopKSAE.from_pretrained(sep_dir)
                payload = build_perm(
                    model=model, cache_dir=cfg.cache.cache_dir,
                    split="train", max_samples=50_000,
                    batch_size=cfg.training.batch_size, device=cfg.training.device,
                )
                save_perm(perm_path, payload)
                logger.info("[perm] saved %s", perm_path)

    # 4) Eval — left as a separate concern; the existing eval scripts under
    #    scripts/run_eval_*.sh point at perm.npz / final/ paths and work.
    #    Future: integrate clean.src.eval modules into a unified runner.
    if stage in ("all", "eval"):
        logger.info("[eval] stage delegated — invoke scripts/run_eval_* with ROOT=%s", out_root)

    # 5) Drop a config snapshot so the run is reproducible.
    with open(out_root / "config.json", "w") as f:
        json.dump({
            "kind": cfg.kind,
            "model": asdict(cfg.model),
            "cache": asdict(cfg.cache),
            "training": asdict(cfg.training),
            "methods": [asdict(m) for m in cfg.methods],
        }, f, indent=2)
    logger.info("[done] cc3m_downstream → %s", out_root)
