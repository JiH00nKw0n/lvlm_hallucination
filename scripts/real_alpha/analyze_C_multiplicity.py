"""Quantify 1:N multiplicity in the co-activation correlation matrix C (rebuttal E1).

Streams paired embeddings through a frozen separated (two-sided) SAE and
accumulates the full-dataset Pearson correlation matrix C between image-side
and text-side dense latents WITHOUT materializing the (N, L) latent tensors
(eval_utils._pearson_C would need ~46 GB for CC3M). Per-batch fp32 partials
are flushed into fp64 CPU accumulators so the 2.8M-sample sums stay exact.

Reports, per threshold tau:
  * per-row (image latent) counts of text latents with C >= tau  → 1:N prevalence
  * per-column (text latent) counts of image latents with C >= tau → N:1 prevalence
  * Hungarian-consistency: fraction of alive rows whose Hungarian partner
    equals the row argmax; top1-top2 margin distribution.

Outputs (under --output-dir):
  * C_stats.npz         — C (fp16), fire/alive masks, n_samples
  * multiplicity.json   — all quantitative stats
  * multiplicity_hist.{pdf,svg}, margin_hist.{pdf,svg}

Usage (server, full CC3M; needs pre-stacked cache for low RSS):
    python scripts/real_alpha/analyze_C_multiplicity.py \
        --ckpt outputs/real_exp_cc3m_s0/separated/ckpt/final \
        --dataset cc3m --cache-dir cache/clip_b32_cc3m \
        --perm outputs/real_exp_cc3m_s0/ours/perm.npz \
        --output-dir outputs/rebuttal_E1/s0

Smoke (local Mac, COCO cache):
    python scripts/real_alpha/analyze_C_multiplicity.py \
        --ckpt outputs/real_exp_cc3m_s0/separated/ckpt/final \
        --dataset coco --cache-dir cache/clip_b32_coco --split train \
        --max-samples 20000 --output-dir /tmp/e1_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLUSH_EVERY = 64  # batches between fp32-partial → fp64-CPU flushes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="separated SAE final ckpt dir")
    p.add_argument("--dataset", choices=["coco", "cc3m", "laion"], default="cc3m")
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--perm", type=str, default=None,
                   help="existing perm.npz for Hungarian-consistency stats")
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = use ALL pairs (full-dataset C); else evenly-spaced subsample")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--alive-min-fires", type=int, default=1)
    p.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def _pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pair_rows(ds) -> tuple[np.ndarray, np.ndarray]:
    """Resolve (image_row, text_row) index arrays for every pair, so we can
    batch-gather straight from the stacked tables instead of per-item
    __getitem__ (which is ~50x slower at CC3M scale)."""
    img_rows = np.empty(len(ds.pairs), dtype=np.int64)
    txt_rows = np.empty(len(ds.pairs), dtype=np.int64)
    for i, (iid, cid) in enumerate(ds.pairs):
        img_rows[i] = ds._image_id_to_row[int(iid)]
        txt_rows[i] = ds._text_key_to_row[ds._text_key_for(iid, cid)]
    return img_rows, txt_rows


@torch.no_grad()
def accumulate_C(two_sae, ds, device: torch.device, batch_size: int,
                 max_samples: int) -> dict:
    """Stream all pairs; return fp64 sufficient statistics + fire counts."""
    two_sae.eval()
    two_sae.to(device)
    L = int(two_sae.image_sae.latent_size)

    img_rows, txt_rows = _pair_rows(ds)
    n_total = len(img_rows)
    if max_samples and max_samples < n_total:
        sel = np.linspace(0, n_total - 1, max_samples, dtype=np.int64)
        img_rows, txt_rows = img_rows[sel], txt_rows[sel]
        n_total = max_samples
    logger.info("accumulating C over %d pairs, L=%d, batch=%d, device=%s",
                n_total, L, batch_size, device)

    # fp64 CPU accumulators (exact at 2.8M scale)
    acc: dict = {k: torch.zeros(L, dtype=torch.float64) for k in
                 ("sum_i", "sum_t", "sumsq_i", "sumsq_t", "fire_i", "fire_t")}
    acc["cross"] = torch.zeros(L, L, dtype=torch.float64)

    # fp32 on-device partials, flushed every FLUSH_EVERY batches
    part = {k: torch.zeros(L, dtype=torch.float32, device=device) for k in
            ("sum_i", "sum_t", "sumsq_i", "sumsq_t", "fire_i", "fire_t")}
    part["cross"] = torch.zeros(L, L, dtype=torch.float32, device=device)

    def _flush():
        for k in acc:
            acc[k] += part[k].cpu().double()
            part[k].zero_()

    img_table, txt_table = ds._image_table, ds._text_table
    n_batches = (n_total + batch_size - 1) // batch_size
    for b in range(n_batches):
        s, e = b * batch_size, min((b + 1) * batch_size, n_total)
        x = img_table[torch.as_tensor(img_rows[s:e])].to(device).unsqueeze(1)
        y = txt_table[torch.as_tensor(txt_rows[s:e])].to(device).unsqueeze(1)
        z_i = two_sae.image_sae(hidden_states=x, return_dense_latents=True).dense_latents.squeeze(1).float()
        z_t = two_sae.text_sae(hidden_states=y, return_dense_latents=True).dense_latents.squeeze(1).float()

        part["sum_i"] += z_i.sum(dim=0)
        part["sum_t"] += z_t.sum(dim=0)
        part["sumsq_i"] += (z_i * z_i).sum(dim=0)
        part["sumsq_t"] += (z_t * z_t).sum(dim=0)
        part["fire_i"] += (z_i != 0).float().sum(dim=0)
        part["fire_t"] += (z_t != 0).float().sum(dim=0)
        part["cross"] += z_i.T @ z_t

        if (b + 1) % FLUSH_EVERY == 0:
            _flush()
        if (b + 1) % 50 == 0 or b == n_batches - 1:
            logger.info("  batch %d/%d", b + 1, n_batches)
    _flush()
    acc["n"] = n_total
    return acc


def pearson_from_stats(acc: dict) -> np.ndarray:
    n = float(acc["n"])
    mean_i = acc["sum_i"] / n
    mean_t = acc["sum_t"] / n
    cov = acc["cross"] / n - torch.outer(mean_i, mean_t)
    var_i = (acc["sumsq_i"] / n - mean_i ** 2).clamp_min(0.0)
    var_t = (acc["sumsq_t"] / n - mean_t ** 2).clamp_min(0.0)
    denom = torch.outer(var_i.sqrt(), var_t.sqrt()).clamp_min(1e-12)
    return (cov / denom).numpy()


def _count_stats(counts: np.ndarray) -> dict:
    """Histogram + headline fractions for an integer count vector."""
    bins = np.bincount(counts, minlength=1)
    return {
        "hist": {str(k): int(v) for k, v in enumerate(bins) if v > 0},
        "frac_0": float((counts == 0).mean()),
        "frac_1": float((counts == 1).mean()),
        "frac_ge_2": float((counts >= 2).mean()),
        "frac_ge_3": float((counts >= 3).mean()),
        "mean": float(counts.mean()),
        "max": int(counts.max()) if counts.size else 0,
        "n": int(counts.size),
    }


def analyze(C: np.ndarray, fire_i: np.ndarray, fire_t: np.ndarray,
            alive_min_fires: int, taus: list[float],
            perm_path: str | None) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alive_i = fire_i >= alive_min_fires
    alive_t = fire_t >= alive_min_fires
    C_alive = C[np.ix_(alive_i, alive_t)]
    out: dict = {
        "alive_image": int(alive_i.sum()), "alive_text": int(alive_t.sum()),
        "latent_size": int(C.shape[0]),
        "alive_min_fires": alive_min_fires,
    }

    per_tau = {}
    for tau in taus:
        row_counts = (C_alive >= tau).sum(axis=1)
        col_counts = (C_alive >= tau).sum(axis=0)
        per_tau[str(tau)] = {
            "row": _count_stats(row_counts),   # image latent → #text partners
            "col": _count_stats(col_counts),   # text latent → #image partners
        }
    out["per_tau"] = per_tau

    # top1 vs top2 margin over alive columns, per alive row
    order = np.sort(C_alive, axis=1)
    top1, top2 = order[:, -1], order[:, -2]
    margin = top1 - top2
    out["row_top1_pct"] = {str(p): float(np.percentile(top1, p)) for p in (5, 25, 50, 75, 95)}
    out["row_margin_pct"] = {str(p): float(np.percentile(margin, p)) for p in (5, 25, 50, 75, 95)}

    if perm_path:
        pz = np.load(perm_path)
        perm = pz["perm"]
        alive_idx_t = np.where(alive_t)[0]
        argmax_alive = alive_idx_t[C_alive.argmax(axis=1)]  # global text idx of row max
        rows_idx = np.where(alive_i)[0]
        hung_partner = perm[rows_idx]
        partner_C = C[rows_idx, hung_partner]
        agree = hung_partner == argmax_alive
        out["hungarian"] = {
            "frac_perm_equals_argmax": float(agree.mean()),
            "n_alive_rows": int(agree.size),
            "frac_partner_C_ge": {
                str(tau): float((partner_C >= tau).mean()) for tau in taus
            },
            # agreement conditioned on partner strength: where the match is
            # strongly co-activating, Hungarian should equal the row argmax
            "agree_given_partner_C_ge": {
                str(tau): {
                    "n": int((partner_C >= tau).sum()),
                    "frac_agree": float(agree[partner_C >= tau].mean())
                    if (partner_C >= tau).any() else None,
                } for tau in taus
            },
        }
    return out, alive_i, alive_t, top1, margin


def plot(out: dict, top1: np.ndarray, margin: np.ndarray,
         taus: list[float], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(taus), figsize=(3.0 * len(taus), 2.4), sharey=True)
    for ax, tau in zip(np.atleast_1d(axes), taus):
        st = out["per_tau"][str(tau)]["row"]["hist"]
        ks = sorted(int(k) for k in st)
        ks_show = [k for k in ks if k <= 8]
        vals = [st[str(k)] for k in ks_show]
        overflow = sum(st[str(k)] for k in ks if k > 8)
        labels = [str(k) for k in ks_show]
        if overflow:
            labels.append(">8"); vals.append(overflow)
        ax.bar(range(len(vals)), vals, color="#4878A8")
        ax.set_xticks(range(len(vals)), labels)
        ax.set_title(rf"$\tau={tau}$", fontsize=9)
        ax.set_xlabel("# partners with $C \\geq \\tau$", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("# image latents", fontsize=8)
    np.atleast_1d(axes)[0].set_yscale("log")
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"multiplicity_hist.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
    axes[0].hist(top1, bins=50, color="#4878A8")
    axes[0].set_xlabel("row max $C$", fontsize=8)
    axes[1].hist(margin, bins=50, color="#B85C5C")
    axes[1].set_xlabel("top1 $-$ top2 margin", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.set_yscale("log")
    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(outdir / f"margin_hist.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = _pick_device(args.device)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("loading separated SAE from %s", args.ckpt)
    model = eval_utils.load_sae(args.ckpt, "separated")
    ds = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, split=args.split)
    logger.info("dataset: %d pairs (split=%s)", len(ds), args.split)

    acc = accumulate_C(model, ds, device, args.batch_size, args.max_samples)
    C = pearson_from_stats(acc)
    fire_i = acc["fire_i"].numpy().astype(np.int64)
    fire_t = acc["fire_t"].numpy().astype(np.int64)

    out, alive_i, alive_t, top1, margin = analyze(
        C, fire_i, fire_t, args.alive_min_fires, args.taus, args.perm)
    out["n_samples"] = int(acc["n"])
    out["ckpt"] = args.ckpt
    out["dataset"] = args.dataset

    np.savez_compressed(
        outdir / "C_stats.npz",
        C=C.astype(np.float16), fire_image=fire_i, fire_text=fire_t,
        alive_image=alive_i, alive_text=alive_t, n_samples=acc["n"],
    )
    with open(outdir / "multiplicity.json", "w") as f:
        json.dump(out, f, indent=2)
    plot(out, top1, margin, args.taus, outdir)

    logger.info("=== E1 headline (alive %d img × %d txt of %d) ===",
                out["alive_image"], out["alive_text"], out["latent_size"])
    for tau in args.taus:
        r = out["per_tau"][str(tau)]["row"]
        logger.info("tau=%.2f  row: 0 partners %.1f%% | exactly 1 %.1f%% | >=2 %.1f%% (max %d)",
                    tau, 100 * r["frac_0"], 100 * r["frac_1"], 100 * r["frac_ge_2"], r["max"])
    if "hungarian" in out:
        logger.info("Hungarian partner == row-argmax: %.1f%% of %d alive rows",
                    100 * out["hungarian"]["frac_perm_equals_argmax"],
                    out["hungarian"]["n_alive_rows"])
    logger.info("wrote %s", outdir)


if __name__ == "__main__":
    main()
