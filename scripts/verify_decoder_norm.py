"""Assert that every W_dec row in an SAE checkpoint has unit L2 norm.

Usage:
    python scripts/verify_decoder_norm.py outputs/real_exp_cc3m/group_sparse/ckpt/final/model.safetensors
    python scripts/verify_decoder_norm.py outputs/real_exp_cc3m/             # walks all model.safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

from safetensors.torch import load_file

TOL = 1e-4


def check(path: Path) -> tuple[bool, list[str]]:
    sd = load_file(str(path))
    msgs = []
    ok = True
    found = False
    for k, v in sd.items():
        if not k.endswith("W_dec"):
            continue
        found = True
        norms = v.norm(dim=1)
        err = (norms - 1.0).abs().max().item()
        status = "OK " if err < TOL else "BAD"
        if err >= TOL:
            ok = False
        msgs.append(
            f"  {status} {k:<25s} min={norms.min():.4f} max={norms.max():.4f} "
            f"mean={norms.mean():.4f} max|n-1|={err:.2e}"
        )
    if not found:
        msgs.append("  WARN no *.W_dec key in checkpoint")
        ok = False
    return ok, msgs


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    if target.is_file():
        paths = [target]
    else:
        paths = sorted(target.rglob("model.safetensors"))
    if not paths:
        print(f"No model.safetensors under {target}")
        return 2

    failed = 0
    for p in paths:
        ok, msgs = check(p)
        print(p)
        for m in msgs:
            print(m)
        if not ok:
            failed += 1
    print(f"\n{len(paths) - failed}/{len(paths)} checkpoints unit-norm (tol={TOL})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
