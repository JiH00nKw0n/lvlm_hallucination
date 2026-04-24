"""YAML-driven real-data SAE experiment driver (v2).

Mirror of ``run_synthetic_v2.py`` for the real-data pipeline:

    python scripts/real_alpha/run_real_v2.py \\
        --config configs/real/cc3m.yaml

Pipeline stages (each one skips its work if the expected artifact exists;
use ``--force`` to re-run and ``--dry-run`` to print commands without
executing):

    1. TRAIN — 4 methods (shared, separated, iso_align, group_sparse) via
       ``train_real_sae.py``. Checkpoints land at
       ``<out>/<method>/ckpt/final``. ``ours`` has ``base_method: separated``
       and reuses that checkpoint, so no retraining.

    2. PERMUTATION — for each ``(eval.dataset)`` whose tasks include
       retrieval or zeroshot_raw, build a Hungarian slot permutation from
       the separated ckpt. Saved to ``<out>/ours/<eval.dataset>/perm.npz``.

    3. EVALUATION — for each ``(method, eval, task)`` triple, run the
       matching eval script and dump JSON under
       ``<out>/<method>/<eval.dataset>/<task>.json``.

    4. TABLE — emit ``<out>/table.{md,tex}`` via
       ``build_real_table.py``.

Each subprocess prints wall-clock duration + cumulative elapsed + ETA for
the remaining steps.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Bootstrap: make src.* importable even when transformers 5.5.3 breaks
# the flex_attention init path (mirrors scripts/real_alpha/_bootstrap.py).
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from src.configs.real_experiment import (  # noqa: E402
    EvalSpec,
    MethodConfig,
    RealExperimentConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_real_v2")


# ---------------------------------------------------------------------------
# Subprocess runner with ETA logging
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One pipeline step. `cost` is a coarse wall-clock estimate (seconds)."""
    tag: str
    cmd: list[str]
    out_artifact: Path
    cost: float
    runner: Callable[[list[str]], int] | None = None


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def _exists(p: Path) -> bool:
    """Checkpoint dir with a saved model, or a non-empty file."""
    if not p.exists():
        return False
    if p.is_dir():
        # HF Trainer saves model.safetensors or pytorch_model.bin inside final/.
        return any((p / name).exists() for name in (
            "model.safetensors", "pytorch_model.bin", "config.json",
        ))
    return p.stat().st_size > 0


def _run_subprocess(cmd: list[str]) -> int:
    # Inherit stdout/stderr so HF Trainer's tqdm ETA flows to the parent log.
    return subprocess.run(cmd, check=False).returncode


def run_steps(steps: list[Step], dry_run: bool, force: bool) -> None:
    total_cost = sum(s.cost for s in steps)
    t0 = time.time()
    cumulative_done = 0.0

    logger.info("Planned steps: %d  (total wall estimate: %s)", len(steps), _fmt_eta(total_cost))
    for i, step in enumerate(steps, start=1):
        remaining = total_cost - cumulative_done
        logger.info(
            "[%d/%d] %s   (step~%s, remaining~%s, elapsed=%s)",
            i, len(steps), step.tag,
            _fmt_eta(step.cost), _fmt_eta(remaining),
            _fmt_eta(time.time() - t0),
        )

        if not force and _exists(step.out_artifact):
            logger.info("  SKIP (exists): %s", step.out_artifact)
            cumulative_done += step.cost
            continue

        logger.info("  CMD: %s", " ".join(step.cmd))
        if dry_run:
            cumulative_done += step.cost
            continue

        step.out_artifact.parent.mkdir(parents=True, exist_ok=True)
        step_start = time.time()
        runner = step.runner or _run_subprocess
        rc = runner(step.cmd)
        dt = time.time() - step_start
        if rc != 0:
            logger.error("  FAILED (rc=%d) after %s: %s", rc, _fmt_eta(dt), step.tag)
            raise SystemExit(rc)
        logger.info("  OK in %s → %s", _fmt_eta(dt), step.out_artifact)
        cumulative_done += step.cost

    logger.info("All %d steps done in %s", len(steps), _fmt_eta(time.time() - t0))


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def _variant_for(method: MethodConfig) -> tuple[str, list[str]]:
    """Map method.name → (--variant, extra-args) for train_real_sae.py."""
    if method.name == "shared":
        return "one_sae", []
    if method.name == "separated":
        return "two_sae", []
    if method.name == "iso_align":
        args = ["--aux-loss", "iso_align"]
        if method.aux_weight > 0:
            args += ["--aux-lambda", str(method.aux_weight)]
        return "aux_sae", args
    if method.name == "group_sparse":
        args = ["--aux-loss", "group_sparse"]
        if method.aux_weight > 0:
            args += ["--aux-lambda", str(method.aux_weight)]
        return "aux_sae", args
    raise ValueError(f"cannot train method {method.name!r} directly")


def _train_cmd(
    cfg: RealExperimentConfig,
    method: MethodConfig,
    ckpt_parent: Path,
    train_script: Path,
) -> list[str]:
    variant, extra = _variant_for(method)
    t = cfg.training
    cmd = [
        sys.executable, str(train_script),
        "--variant", variant,
        "--dataset", cfg.data.dataset,
        "--cache-dir", cfg.data.cache_dir,
        "--output-dir", str(ckpt_parent),
        "--latent", str(t.latent_size),
        "--k", str(t.k),
        "--hidden-size", str(t.hidden_size),
        "--epochs", str(t.num_epochs),
        "--batch-size", str(t.batch_size),
        "--lr", str(t.lr),
        "--warmup-ratio", str(t.warmup_ratio),
        "--weight-decay", str(t.weight_decay),
        "--max-grad-norm", str(t.max_grad_norm),
        "--seed", str(t.seed),
        "--dataloader-num-workers", str(t.dataloader_num_workers),
    ]
    if cfg.data.dataset == "cc3m" and cfg.data.eval_samples > 0:
        cmd += ["--eval-samples", str(cfg.data.eval_samples)]
    if cfg.data.dataset == "imagenet":
        cmd += ["--max-per-class", str(cfg.data.max_per_class)]
    cmd += extra
    return cmd


def _perm_cmd(
    separated_ckpt: Path,
    ev: EvalSpec,
    perm_path: Path,
    script: Path,
) -> list[str]:
    return [
        sys.executable, str(script),
        "--ckpt", str(separated_ckpt),
        "--dataset", ev.dataset,
        "--cache-dir", ev.cache_dir,
        "--output", str(perm_path),
    ]


def _recon_cmd(
    ckpt: Path, method_tag: str, ev: EvalSpec, out_json: Path, script: Path,
) -> list[str]:
    # For the recon task `ours` uses the separated ckpt but reports as separated
    # (the Hungarian permutation is a no-op on reconstruction).
    # We still call eval_recon with method=ours so eval_utils.load_sae returns
    # TwoSidedTopKSAE consistently.
    return [
        sys.executable, str(script),
        "--ckpt", str(ckpt),
        "--method", method_tag,
        "--dataset", ev.dataset,
        "--cache-dir", ev.cache_dir,
        "--split", ev.split,
        "--output", str(out_json),
    ]


def _retrieval_cmd(
    ckpt: Path, method_tag: str, ev: EvalSpec, out_json: Path,
    perm: Path | None, script: Path,
) -> list[str]:
    cmd = [
        sys.executable, str(script),
        "--ckpt", str(ckpt),
        "--method", method_tag,
        "--cache-dir", ev.cache_dir,
        "--split", ev.split,
        "--output", str(out_json),
    ]
    if perm is not None:
        cmd += ["--perm", str(perm)]
    return cmd


def _zs_cmd(
    ckpt: Path, method_tag: str, ev: EvalSpec, out_json: Path,
    perm: Path | None, script: Path,
) -> list[str]:
    cmd = [
        sys.executable, str(script),
        "--ckpt", str(ckpt),
        "--method", method_tag,
        "--cache-dir", ev.cache_dir,
        "--output", str(out_json),
    ]
    if perm is not None:
        cmd += ["--perm", str(perm)]
    return cmd


def _dead_cmd(
    ckpt: Path, method_tag: str, ev: EvalSpec, out_json: Path, script: Path,
) -> list[str]:
    return [
        sys.executable, str(script),
        "--ckpt", str(ckpt),
        "--method", method_tag,
        "--dataset", ev.dataset,
        "--cache-dir", ev.cache_dir,
        "--output", str(out_json),
    ]


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------


def _method_tag(method: MethodConfig) -> str:
    """eval_utils.load_sae method token. `iso_align`/`group_sparse` map to `aux`."""
    if method.name in ("iso_align", "group_sparse"):
        return "aux"
    if method.name == "shared":
        return "shared"
    if method.name == "separated":
        return "separated"
    if method.name == "ours":
        return "ours"
    raise ValueError(method.name)


def _resolve_ckpt(method: MethodConfig, out_root: Path) -> Path:
    base = method.base_method or method.name
    return out_root / base / "ckpt" / "final"


def build_steps(
    cfg: RealExperimentConfig,
    script_dir: Path,
    out_root: Path,
) -> list[Step]:
    steps: list[Step] = []

    train_script = script_dir / "train_real_sae.py"
    perm_script = script_dir / "build_hungarian_perm.py"
    recon_script = script_dir / "eval_recon_downstream.py"
    retrieval_script = script_dir / "eval_coco_retrieval.py"
    zs_script = script_dir / "eval_imagenet_zeroshot.py"
    dead_script = script_dir / "eval_dead_latents.py"

    # Rough costs (seconds) — tuned for a 10GB GPU, CC3M-scale train set.
    COST_TRAIN = 2000.0   # ~35min per method on CC3M k=32 L=8192 30ep
    COST_PERM = 180.0
    COST_EVAL_RECON = 60.0
    COST_EVAL_RETR = 180.0
    COST_EVAL_ZS = 300.0
    COST_EVAL_DEAD = 120.0

    # 1. TRAIN
    for m in cfg.methods:
        if m.base_method:
            continue
        ckpt_parent = out_root / m.name / "ckpt"
        final_ckpt = ckpt_parent / "final"
        steps.append(Step(
            tag=f"train[{m.name}]",
            cmd=_train_cmd(cfg, m, ckpt_parent, train_script),
            out_artifact=final_ckpt,
            cost=COST_TRAIN,
        ))

    # 2. PERM — only for eval.datasets that need slot alignment.
    has_ours = any(m.name == "ours" for m in cfg.methods)
    perm_targets: dict[str, EvalSpec] = {}
    if has_ours:
        for ev in cfg.evaluations:
            needs_perm = any(t in ("retrieval", "zeroshot_raw") for t in ev.tasks)
            if needs_perm and ev.dataset not in perm_targets:
                perm_targets[ev.dataset] = ev

    sep_ckpt = out_root / "separated" / "ckpt" / "final"
    for ev_ds, ev in perm_targets.items():
        perm_path = out_root / "ours" / ev_ds / "perm.npz"
        steps.append(Step(
            tag=f"perm[ours → {ev_ds}]",
            cmd=_perm_cmd(sep_ckpt, ev, perm_path, perm_script),
            out_artifact=perm_path,
            cost=COST_PERM,
        ))

    # 3. EVAL
    for ev in cfg.evaluations:
        for m in cfg.methods:
            ckpt = _resolve_ckpt(m, out_root)
            out_dir = out_root / m.name / ev.dataset
            method_tag = _method_tag(m)
            perm = None
            if m.name == "ours" and ev.dataset in perm_targets:
                perm = out_root / "ours" / ev.dataset / "perm.npz"
            for task in ev.tasks:
                out_json = out_dir / f"{task}.json"
                if task == "recon":
                    cmd = _recon_cmd(ckpt, method_tag, ev, out_json, recon_script)
                    cost = COST_EVAL_RECON
                elif task == "retrieval":
                    cmd = _retrieval_cmd(ckpt, method_tag, ev, out_json, perm, retrieval_script)
                    cost = COST_EVAL_RETR
                elif task == "zeroshot_raw":
                    cmd = _zs_cmd(ckpt, method_tag, ev, out_json, perm, zs_script)
                    cost = COST_EVAL_ZS
                elif task == "dead_latents":
                    cmd = _dead_cmd(ckpt, method_tag, ev, out_json, dead_script)
                    cost = COST_EVAL_DEAD
                else:
                    raise ValueError(f"unknown task {task!r}")
                steps.append(Step(
                    tag=f"eval[{m.name}/{ev.dataset}/{task}]",
                    cmd=cmd,
                    out_artifact=out_json,
                    cost=cost,
                ))

    # 4. TABLE — always emit (cheap)
    table_md = out_root / "table.md"
    table_script = script_dir / "build_real_table.py"
    steps.append(Step(
        tag="table",
        cmd=[sys.executable, str(table_script), "--config", os.environ.get("_RUN_REAL_V2_CONFIG", ""),
             "--out-root", str(out_root)],
        out_artifact=table_md,
        cost=5.0,
    ))
    return steps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands only; no execution.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run every step even if artifact already exists.")
    ap.add_argument("--skip-table", action="store_true",
                    help="Do not build the table at the end.")
    args = ap.parse_args()

    config_path = str(Path(args.config).resolve())
    os.environ["_RUN_REAL_V2_CONFIG"] = config_path
    cfg = RealExperimentConfig.from_yaml(config_path)
    out_root = Path(cfg.output.root)
    script_dir = Path(__file__).resolve().parent

    logger.info("config=%s", config_path)
    logger.info("output_root=%s  (exists=%s)", out_root, out_root.exists())
    logger.info(
        "training source=%s  cache=%s  k=%d  L=%d  epochs=%d",
        cfg.data.dataset, cfg.data.cache_dir,
        cfg.training.k, cfg.training.latent_size, cfg.training.num_epochs,
    )
    logger.info("methods=%s", [m.name for m in cfg.methods])
    logger.info("evaluations=%s",
                [(ev.dataset, ev.tasks) for ev in cfg.evaluations])

    steps = build_steps(cfg, script_dir, out_root)
    if args.skip_table:
        steps = [s for s in steps if s.tag != "table"]

    run_steps(steps, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
