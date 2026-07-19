"""Build soft (Sinkhorn) and alternative hard text→image alignments (rebuttal E6).

Answers the reviewer question "why strict 1:1 Hungarian?" by producing, from
the SAME co-activation protocol as build_hungarian_perm.py (50k training
subsample, alive masking):

  * soft_T_eps<eps>.npz  — entropic-OT transport plan T (fp16, L×L, dead
    rows/cols zero) + a hardened "perm" (Hungarian on -T) so the same file
    also works as a drop-in --perm.
  * perm_greedy.npz      — greedy max-correlation matching ("perm")
  * perm_dec_cos.npz     — Hungarian on decoder-cosine (weight-space only,
    no co-activation) ("perm")
  * perm_hungarian.npz   — the reference Hungarian on C ("perm"), for
    self-containment (should match the shipped ours/perm.npz)

At eval time the soft plan is applied as z'_T = row_norm(T) @ z_T via
eval_utils.encode_text(soft_T=...); hard perms go through --perm unchanged.

Usage:
    python scripts/real_alpha/build_soft_assignment.py \
        --ckpt outputs/real_exp_cc3m_s0/separated/ckpt/final \
        --dataset cc3m --cache-dir cache/clip_b32_cc3m \
        --output-dir outputs/rebuttal_E6/s0 --eps 0.01 0.05 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset", choices=["coco", "cc3m", "laion"], default="cc3m")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--alive-min-fires", type=int, default=1)
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--sinkhorn-iters", type=int, default=300)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def compute_C_and_latents(two_sae, ds, device, max_samples, batch_size):
    """Mirror eval_utils.build_perm's protocol, but return C + fire counts."""
    n = min(len(ds), max_samples)
    indices = np.linspace(0, len(ds) - 1, n, dtype=np.int64)
    img = torch.stack([ds[int(i)]["image_embeds"] for i in indices], dim=0)
    txt = torch.stack([ds[int(i)]["text_embeds"] for i in indices], dim=0)
    z_i = eval_utils._stream_dense_latents(two_sae.image_sae, img, batch_size, device)
    z_t = eval_utils._stream_dense_latents(two_sae.text_sae, txt, batch_size, device)
    fire_i = (z_i != 0).sum(dim=0).numpy().astype(np.int64)
    fire_t = (z_t != 0).sum(dim=0).numpy().astype(np.int64)
    C = eval_utils._pearson_C(z_i, z_t).astype(np.float64)
    return C, fire_i, fire_t


def hungarian_perm(C_masked: np.ndarray) -> np.ndarray:
    row_ind, col_ind = linear_sum_assignment(-C_masked)
    perm = np.zeros_like(col_ind)
    perm[row_ind] = col_ind
    return perm.astype(np.int64)


def greedy_perm(C: np.ndarray, alive_i: np.ndarray, alive_t: np.ndarray) -> np.ndarray:
    """Greedy max-correlation 1:1 matching on the alive×alive submatrix.
    Dead slots map to themselves (their latents are all-zero anyway)."""
    L = C.shape[0]
    perm = np.arange(L, dtype=np.int64)
    rows = np.where(alive_i)[0]
    cols = np.where(alive_t)[0]
    sub = C[np.ix_(rows, cols)]
    order = np.argsort(sub, axis=None)[::-1]
    used_r = np.zeros(len(rows), dtype=bool)
    used_c = np.zeros(len(cols), dtype=bool)
    n_matched = 0
    target = min(len(rows), len(cols))
    for flat in order:
        r, c = divmod(int(flat), len(cols))
        if used_r[r] or used_c[c]:
            continue
        perm[rows[r]] = cols[c]
        used_r[r] = used_c[c] = True
        n_matched += 1
        if n_matched >= target:
            break
    return perm


def dec_cos_perm(two_sae, alive_i: np.ndarray, alive_t: np.ndarray) -> np.ndarray:
    """Hungarian on decoder-direction cosine (no co-activation signal)."""
    Wi = two_sae.image_sae.W_dec.detach().float().cpu().numpy()
    Wt = two_sae.text_sae.W_dec.detach().float().cpu().numpy()
    Wi = Wi / np.clip(np.linalg.norm(Wi, axis=1, keepdims=True), 1e-12, None)
    Wt = Wt / np.clip(np.linalg.norm(Wt, axis=1, keepdims=True), 1e-12, None)
    S = (Wi @ Wt.T).astype(np.float64)
    BIG_NEG = -1e9
    S[~alive_i, :] = BIG_NEG
    S[:, ~alive_t] = BIG_NEG
    return hungarian_perm(S)


def sinkhorn_T(C: np.ndarray, alive_i: np.ndarray, alive_t: np.ndarray,
               eps: float, iters: int, device: torch.device) -> np.ndarray:
    """Log-domain entropic OT with uniform marginals on alive×alive; returns
    a full L×L plan (dead rows/cols zero), scaled so alive rows sum to 1."""
    rows = np.where(alive_i)[0]
    cols = np.where(alive_t)[0]
    logK = torch.as_tensor(C[np.ix_(rows, cols)] / eps, dtype=torch.float64, device=device)
    n_r, n_c = logK.shape
    log_mu = -np.log(n_r) * torch.ones(n_r, dtype=torch.float64, device=device)
    log_nu = -np.log(n_c) * torch.ones(n_c, dtype=torch.float64, device=device)
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(logK + v[None, :], dim=1)
        v = log_nu - torch.logsumexp(logK + u[:, None], dim=0)
    T_sub = torch.exp(logK + u[:, None] + v[None, :])
    # row-normalize so each alive image slot's mixture sums to 1
    T_sub = T_sub / T_sub.sum(dim=1, keepdim=True).clamp_min(1e-30)
    L = C.shape[0]
    T = np.zeros((L, L), dtype=np.float32)
    T[np.ix_(rows, cols)] = T_sub.cpu().float().numpy()
    return T


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    model = eval_utils.load_sae(args.ckpt, "separated")
    ds = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, split=args.split)
    logger.info("dataset=%d pairs; computing C on %d subsample", len(ds), args.max_samples)
    C, fire_i, fire_t = compute_C_and_latents(model, ds, device, args.max_samples, args.batch_size)
    alive_i = fire_i >= args.alive_min_fires
    alive_t = fire_t >= args.alive_min_fires
    logger.info("alive: img %d/%d, txt %d/%d", alive_i.sum(), len(alive_i),
                alive_t.sum(), len(alive_t))

    BIG_NEG = -1e9
    C_masked = C.copy()
    C_masked[~alive_i, :] = BIG_NEG
    C_masked[:, ~alive_t] = BIG_NEG

    common = dict(alive_image=alive_i, alive_text=alive_t,
                  fire_count_image=fire_i, fire_count_text=fire_t)

    p_h = hungarian_perm(C_masked)
    np.savez(outdir / "perm_hungarian.npz", perm=p_h, **common)
    logger.info("hungarian done")

    p_g = greedy_perm(C, alive_i, alive_t)
    np.savez(outdir / "perm_greedy.npz", perm=p_g, **common)
    logger.info("greedy done (agree with hungarian on %.1f%% of alive rows)",
                100 * float((p_g[alive_i] == p_h[alive_i]).mean()))

    p_d = dec_cos_perm(model, alive_i, alive_t)
    np.savez(outdir / "perm_dec_cos.npz", perm=p_d, **common)
    logger.info("dec_cos done (agree with hungarian on %.1f%% of alive rows)",
                100 * float((p_d[alive_i] == p_h[alive_i]).mean()))

    sink_device = device if device.type == "cuda" else torch.device("cpu")  # fp64: no MPS
    for eps in args.eps:
        T = sinkhorn_T(C, alive_i, alive_t, eps, args.sinkhorn_iters, sink_device)
        hardened = hungarian_perm(np.where(T > 0, T, BIG_NEG))
        agree = float((hardened[alive_i] == p_h[alive_i]).mean())
        np.savez(outdir / f"soft_T_eps{eps}.npz",
                 T=T.astype(np.float16), perm=hardened, eps=eps, **common)
        logger.info("sinkhorn eps=%g done (hardened agrees with hungarian: %.1f%%)",
                    eps, 100 * agree)
    logger.info("wrote %s", outdir)


if __name__ == "__main__":
    main()
