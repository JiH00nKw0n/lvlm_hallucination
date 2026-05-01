"""Monosemanticity evaluation on ImageNet-1K val for CC3M-trained SAE variants.

Three metrics, reported under three dead-latent filters
(alive ≥ 1, well-supp ≥ 50, well-supp ≥ 200):

    M1 — Pach et al. (2025) MonoSemanticity score
         per-latent image-pair similarity weighted by activations
         (DINOv2 ViT-B as external image encoder E)

    M2 — Härle et al. (2025) Feature Monosemanticity score (FMS)
         tree-based concept isolation; matches the reference repo
         (ml-research/measuring-and-guiding-monosemanticity) protocol exactly:
           - acc_0 = depth-1 stump in-sample accuracy
           - local: iterative root-feature cut (5x), each step refits a stump
           - global: from a single max_depth=None tree, in-sample accuracy at
             each depth d=1..max_depth; FMS_global = 1 - sum(acc_d - acc_1)/(N-1)
           - local/global clipped to [0, 1]

    M3 — Class Concentration Score (CCS)
         per-class mean activation distribution top-1 share + 1-H/Hmax

(Absorption rate from Chanin et al. is intentionally omitted — too
heuristic for our setting.)

Usage:
    python scripts/real_alpha/eval_monosemanticity.py \
        --variant shared --method shared \
        --ckpt outputs/real_exp_cc3m/shared/ckpt/final \
        --clip-cache cache/clip_b32_imagenet \
        --dino-cache cache/dinov2_b14_imagenet \
        --out outputs/real_exp_cc3m/monosemanticity/shared \
        --metrics ms,fms,ccs
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys as _sys
import time
import warnings
from pathlib import Path as _Path

# StratifiedKFold complains when --max-samples shrinks per-class counts below
# n_splits — irrelevant on the full 50K val (50 samples/class). Silenced for log cleanliness.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from tqdm import tqdm  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Heartbeat helper — emits structured progress lines at intervals
# ----------------------------------------------------------------------

class Heartbeat:
    def __init__(self, stage: str, variant: str, total: int, every: int):
        self.stage = stage
        self.variant = variant
        self.total = total
        self.every = max(1, every)
        self.t0 = time.time()
        self.last = 0

    def step(self, n: int, extra: str = "") -> None:
        if n - self.last < self.every and n != self.total:
            return
        self.last = n
        elapsed = time.time() - self.t0
        rate = n / elapsed if elapsed > 0 else 0.0
        eta = (self.total - n) / rate if rate > 0 else float("inf")
        logger.info(
            "[heartbeat] stage=%s variant=%s progress=%d/%d elapsed=%s rate=%.2f/s eta=%s%s",
            self.stage, self.variant, n, self.total,
            _fmt_dur(elapsed), rate, _fmt_dur(eta) if eta != float("inf") else "inf",
            (" " + extra) if extra else "",
        )


def _fmt_dur(s: float) -> str:
    if s == float("inf"):
        return "inf"
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, required=True,
                   help="Run-dir variant name (e.g. shared, separated, iso_align, group_sparse)")
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae"], required=True,
                   help="eval_utils.load_sae() method dispatcher")
    p.add_argument("--ckpt", type=str, required=True, help="Path to ckpt/final dir")
    p.add_argument("--clip-cache", type=str, required=True)
    p.add_argument("--dino-cache", type=str, default=None,
                   help="Cache dir with DINOv2 image_embeddings.pt; required for --metrics ms")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--metrics", type=str, default="ms,fms,ccs",
                   help="Comma-separated subset of {ms,fms,ccs}")
    p.add_argument("--well-supp-thresholds", type=str, default="50,200")
    p.add_argument("--ccs-max-fire-frac", type=float, default=1.0,
                   help="Drop latents firing on more than this fraction of N images "
                        "before computing CCS (1.0 = no filter)")
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--max-samples", type=int, default=0,
                   help="If > 0, cap val samples (for local smoke test)")
    p.add_argument("--fms-local-max-p", type=int, default=5,
                   help="Number of iterative root-feature cuts (paper FMS@p, p in {1,...,max_p})")
    p.add_argument("--fms-neg-per-class", type=int, default=50)
    p.add_argument("--encode-batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------
# Helpers — encoding / loading
# ----------------------------------------------------------------------

def _pick_image_sae(model, method: str):
    if method in ("shared", "aux"):
        return model
    if method == "vl_sae":
        return model
    return model.image_sae


def _load_clip_val(cache_dir: str) -> tuple[torch.Tensor, np.ndarray, list[int]]:
    """Returns (img_embeds[N,D], labels[N], image_ids[N]) — order matches splits['val']."""
    ds = eval_utils.load_pair_dataset(cache_dir, "imagenet", "val")
    N = len(ds)
    img = torch.stack([ds[i]["image_embeds"] for i in range(N)], dim=0)
    labels = np.array([int(ds.pairs[i][1]) for i in range(N)], dtype=np.int64)
    image_ids = [int(ds.pairs[i][0]) for i in range(N)]
    return img, labels, image_ids


def _load_dino_val(dino_cache: str, image_ids: list[int]) -> torch.Tensor:
    p = _Path(dino_cache) / "image_embeddings.pt"
    raw = torch.load(p, map_location="cpu")
    raw = {int(k): v for k, v in raw.items()}
    out = torch.stack([raw[iid] for iid in image_ids], dim=0).float()
    out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


# ----------------------------------------------------------------------
# Stage 0: encode + fire counts
# ----------------------------------------------------------------------

@torch.no_grad()
def stage_encode(args: argparse.Namespace, model, img: torch.Tensor, device: torch.device) -> torch.Tensor:
    image_sae = _pick_image_sae(model, args.method)
    image_sae.eval(); image_sae.to(device)
    if args.method == "vl_sae":
        L = int(image_sae.cfg.latent_size)
    else:
        L = int(image_sae.latent_size)
    N = img.shape[0]
    out = torch.empty(N, L, dtype=torch.float32)
    for s in tqdm(range(0, N, args.encode_batch_size),
                  desc=f"encode/{args.variant}", mininterval=2.0):
        chunk = img[s:s + args.encode_batch_size].to(device)
        if args.method == "vl_sae":
            z = image_sae.encode(chunk)
        else:
            z = image_sae(hidden_states=chunk.unsqueeze(1),
                          return_dense_latents=True).dense_latents.squeeze(1)
        out[s:s + chunk.shape[0]] = z.float().cpu()
    return out


def fire_counts(z: torch.Tensor, thresholds: list[int]) -> dict:
    fire = (z != 0).sum(dim=0).cpu().numpy().astype(np.int64)
    L = int(z.shape[1])
    alive_sets = {f"A_{t}": int((fire >= t).sum()) for t in [1] + thresholds}
    return {"L": L, "fire_count": fire.tolist(), **alive_sets}


# ----------------------------------------------------------------------
# M1 — MS score
# ----------------------------------------------------------------------

def compute_ms(
    z: torch.Tensor,           # (N, L)
    dino_norm: torch.Tensor,   # (N, D_dino), L2-normalized
    alive_idx: np.ndarray,
    variant: str,
    device: torch.device,
    heartbeat_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MS_k for each k in alive_idx. Returns:
        ms (|alive|,), fire (|alive|,), top16_ids (|alive|, 16) — image-position indices

    Math (Pach Eq. 9, only firing samples contribute since ã=0 for non-firing):
        ã = a / max_n a               (min over all N is 0 for TopK)
        MS_k = (ã^T S ã − sum ã²) / (sum ã)² − sum ã² )
             = off-diagonal r-weighted similarity / off-diagonal r-mass

    Latents with fewer than 2 firings get ms=NaN.
    """
    alive = list(map(int, alive_idx))
    n_alive = len(alive)
    ms = np.full(n_alive, np.nan, dtype=np.float64)
    fire = np.zeros(n_alive, dtype=np.int64)
    top16 = np.full((n_alive, 16), -1, dtype=np.int64)

    dino_dev = dino_norm.to(device)
    z_dev_T = z.to(device).T  # (L, N)

    hb = Heartbeat("ms", variant, n_alive, heartbeat_every)
    pbar = tqdm(range(n_alive), desc=f"ms/{variant}", mininterval=2.0, smoothing=0.1)
    running_sum = 0.0
    running_n = 0
    for i in pbar:
        k = alive[i]
        a_full = z_dev_T[k]                               # (N,)
        mask = a_full > 0
        f = int(mask.sum().item())
        fire[i] = f
        if f < 2:
            hb.step(i + 1)
            continue
        firing = mask.nonzero(as_tuple=False).squeeze(-1)  # (f,)
        a_F = a_full[firing]                                # (f,)
        a_max = a_F.max().clamp_min(1e-12)
        a_tilde = a_F / a_max                                # (f,)

        e_F = dino_dev[firing]                               # (f, D)
        # Closed form: MS_k = (ã^T S ã − ã^T ã) / ((sum ã)² − ã^T ã)
        # ã^T S ã = ã^T (E E^T) ã = ||E^T ã||²
        proj = a_tilde @ e_F                                 # (D,)
        ata_S_at = float((proj * proj).sum().item())
        a_norm2 = float((a_tilde * a_tilde).sum().item())
        a_sum = float(a_tilde.sum().item())
        num = ata_S_at - a_norm2
        den = a_sum * a_sum - a_norm2
        if den > 1e-30:
            ms[i] = num / den
            running_sum += ms[i]; running_n += 1

        topk_n = min(16, f)
        _, idx = torch.topk(a_F, topk_n)
        top16[i, :topk_n] = firing[idx].cpu().numpy()
        if i % 200 == 0 and running_n > 0:
            pbar.set_postfix(ms_running=running_sum / running_n)
        hb.step(i + 1)

    return ms, fire, top16


# ----------------------------------------------------------------------
# M2 — FMS score (paper-faithful protocol)
# ----------------------------------------------------------------------
# Reference: ml-research/measuring-and-guiding-monosemanticity
#   - utils/create_trees.py::cut_RTP_tree (iterative root-feature cut)
#   - utils/tree_loader.py::get_tree_stats (depth-by-depth in-sample accuracy)
#   - eval/FMS_Scores.ipynb (aggregation)


def _stump_in_sample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, int]:
    """Fit max_depth=1 tree, return (in-sample accuracy, root_feature_idx)."""
    clf = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=seed)
    clf.fit(X, y)
    return float(clf.score(X, y)), int(clf.tree_.feature[0])


def compute_fms(
    z: torch.Tensor,                   # (N, L)
    labels: np.ndarray,                # (N,)
    alive_idx: np.ndarray,
    n_classes: int,
    variant: str,
    local_max_p: int,
    neg_per_class: int,
    seed: int,
) -> list[dict]:
    z_alive = z[:, torch.as_tensor(alive_idx, dtype=torch.long)].cpu().numpy()
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    hb = Heartbeat("fms", variant, n_classes, every=50)
    pbar = tqdm(range(n_classes), desc=f"fms/{variant}", mininterval=2.0, smoothing=0.1)
    for c in pbar:
        pos_idx = np.where(labels == c)[0]
        if len(pos_idx) == 0:
            empty = {"class_idx": c, "acc_0": np.nan, "fms_global": np.nan,
                     "max_depth": np.nan, "cut_features": "", "acc_by_depth": ""}
            for p in range(1, local_max_p + 1):
                empty[f"acc_p{p}"] = np.nan
                empty[f"fms_local_p{p}"] = np.nan
            rows.append(empty)
            hb.step(c + 1)
            continue
        # negatives — balanced random sample from other classes
        neg_pool = np.where(labels != c)[0]
        neg_size = min(neg_per_class, len(neg_pool))
        neg_idx = rng.choice(neg_pool, size=neg_size, replace=False)
        n_pos = min(neg_per_class, len(pos_idx))
        sel = np.concatenate([pos_idx[:n_pos], neg_idx])
        y = np.concatenate([np.ones(n_pos, dtype=np.int64),
                            np.zeros(neg_size, dtype=np.int64)])
        X = z_alive[sel].copy()  # copy: iterative cut mutates this

        # === acc_0 = depth-1 stump in-sample accuracy ===
        acc_0, _ = _stump_in_sample(X, y, seed)

        # === Global: single max_depth=None tree, accuracy at each depth d ===
        full = DecisionTreeClassifier(criterion="gini", random_state=seed)
        full.fit(X, y)
        max_d = int(full.tree_.max_depth)
        # Sklearn builds tree greedy top-down → max_depth=d retrains give
        # identical depth-d prefixes, so per-depth in-sample accuracy is recovered
        # by training d-bounded trees. Reference does in-tree pruning (same result).
        acc_by_depth = []
        for d in range(1, max_d + 1):
            clf_d = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=seed)
            clf_d.fit(X, y)
            acc_by_depth.append(float(clf_d.score(X, y)))
        if max_d <= 1:
            fms_global = 1.0  # no deeper info possible (tree saturated at depth 1)
        else:
            extra = sum(acc_by_depth[d - 1] - acc_by_depth[0] for d in range(2, max_d + 1))
            fms_global = 1.0 - extra / (max_d - 1)
        fms_global = float(np.clip(fms_global, 0.0, 1.0))

        # === Local: iterative root-feature cut (max_depth=1 stump per step) ===
        # Step p: fit stump on current X → record acc_p, root feat → zero out feat.
        # acc at p-th iteration's new root = stump in-sample acc on (p-1)-cut data.
        X_iter = X.copy()
        cut_features: list[int] = []
        acc_p: dict[int, float] = {}
        for p in range(1, local_max_p + 1):
            ap, root_feat = _stump_in_sample(X_iter, y, seed)
            acc_p[p] = ap
            cut_features.append(root_feat)
            X_iter[:, root_feat] = 0.0
            if all(np.all(X_iter[:, f] == 0) for f in cut_features):
                # nothing left to split on; remaining acc_p stay at majority baseline
                pass

        # Compute fms_local for all p in [1..local_max_p]
        fms_local = {p: float(np.clip(2.0 * (acc_0 - acc_p[p]), 0.0, 1.0))
                     for p in acc_p}

        row = {
            "class_idx": int(c),
            "acc_0": float(acc_0),
            "fms_global": fms_global,
            "max_depth": int(max_d),
            "cut_features": ";".join(str(int(alive_idx[f])) for f in cut_features),
            "acc_by_depth": ";".join(f"{a:.4f}" for a in acc_by_depth),
        }
        for p in range(1, local_max_p + 1):
            row[f"acc_p{p}"] = float(acc_p[p])
            row[f"fms_local_p{p}"] = fms_local[p]
        rows.append(row)
        if c % 100 == 0 and rows:
            running = float(np.mean([r["acc_0"] for r in rows
                                     if not (isinstance(r["acc_0"], float) and np.isnan(r["acc_0"]))]))
            pbar.set_postfix(acc_0_running=running)
        hb.step(c + 1)
    return rows


# ----------------------------------------------------------------------
# M3 (renumbered M4 in plan) — CCS
# ----------------------------------------------------------------------

def compute_ccs(
    z: torch.Tensor,                   # (N, L)
    labels: np.ndarray,
    alive_idx: np.ndarray,
    n_classes: int,
) -> list[dict]:
    z_alive = z[:, torch.as_tensor(alive_idx, dtype=torch.long)]
    A = z_alive.shape[1]
    log_A = float(np.log(max(A, 2)))
    rows: list[dict] = []
    for c in range(n_classes):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            rows.append({"class_idx": c, "ccs_top1": np.nan, "ccs_entropy": np.nan,
                         "top1_latent": -1, "top3_latents": ""})
            continue
        bar_a = z_alive[idx].mean(dim=0).numpy()       # (A,)
        s = bar_a.sum()
        if s <= 0:
            rows.append({"class_idx": c, "ccs_top1": 0.0, "ccs_entropy": 0.0,
                         "top1_latent": -1, "top3_latents": ""})
            continue
        p = bar_a / s
        # top-3 by share (descending)
        top3 = np.argsort(-bar_a)[:3]
        top1_share = float(p[top3[0]])
        nz = p[p > 0]
        H = float(-(nz * np.log(nz)).sum())
        ccs_h = 1.0 - H / max(log_A, 1e-12)
        # Map back to global latent indices
        top3_global = [int(alive_idx[k]) for k in top3]
        rows.append({
            "class_idx": int(c),
            "ccs_top1": top1_share,
            "ccs_entropy": ccs_h,
            "top1_latent": top3_global[0],
            "top3_latents": ";".join(str(k) for k in top3_global),
        })
    return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _write_csv(path: _Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = _Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = set(s.strip() for s in args.metrics.split(",") if s.strip())
    well_supp = [int(t) for t in args.well_supp_thresholds.split(",")]

    # Load
    logger.info("[stage=load] BEGIN variant=%s method=%s ckpt=%s",
                args.variant, args.method, args.ckpt)
    t0 = time.time()
    model = eval_utils.load_sae(args.ckpt, args.method)
    img, labels, image_ids = _load_clip_val(args.clip_cache)
    if args.max_samples > 0:
        img = img[: args.max_samples]
        labels = labels[: args.max_samples]
        image_ids = image_ids[: args.max_samples]
    N, D = img.shape
    logger.info("[stage=load] DONE variant=%s N=%d D=%d dt=%s",
                args.variant, N, D, _fmt_dur(time.time() - t0))

    # Encode
    logger.info("[stage=encode] BEGIN variant=%s", args.variant)
    t0 = time.time()
    z = stage_encode(args, model, img, device)
    logger.info("[stage=encode] DONE variant=%s shape=%s dt=%s",
                args.variant, tuple(z.shape), _fmt_dur(time.time() - t0))

    # Fire counts
    logger.info("[stage=fire_count] BEGIN variant=%s", args.variant)
    t0 = time.time()
    fc = fire_counts(z, well_supp)
    fire = np.array(fc["fire_count"], dtype=np.int64)
    alive = np.where(fire >= 1)[0]
    fc_payload = {
        "L": fc["L"], "n_alive": int(len(alive)),
        **{k: v for k, v in fc.items() if k.startswith("A_")},
        "well_supp_thresholds": well_supp,
        "alive_idx": alive.tolist(),
        "fire_count": fire.tolist(),
    }
    with open(out / "fire_counts.json", "w") as f:
        json.dump(fc_payload, f)
    logger.info("[stage=fire_count] DONE variant=%s alive=%d/%d dt=%s",
                args.variant, len(alive), fc["L"], _fmt_dur(time.time() - t0))

    # M1 — MS
    if "ms" in metrics:
        if not args.dino_cache:
            raise SystemExit("--dino-cache required for --metrics ms")
        logger.info("[stage=ms] BEGIN variant=%s alive=%d", args.variant, len(alive))
        t0 = time.time()
        dino = _load_dino_val(args.dino_cache, image_ids)
        logger.info("[stage=ms] dino shape=%s", tuple(dino.shape))
        ms, fire_alive, top16 = compute_ms(z, dino, alive, args.variant, device)

        # Per-latent CSV
        rows_ms = []
        for i, k in enumerate(alive):
            rows_ms.append({
                "latent_idx": int(k),
                "fire_count": int(fire_alive[i]),
                "ms_score": float(ms[i]) if not np.isnan(ms[i]) else "",
                "top16_imgs": ";".join(str(int(x)) for x in top16[i] if x >= 0),
            })
        _write_csv(out / "ms_per_latent.csv", rows_ms,
                   ["latent_idx", "fire_count", "ms_score", "top16_imgs"])

        def agg(mask: np.ndarray) -> dict:
            sel = ms[mask]
            sel = sel[~np.isnan(sel)]
            if len(sel) == 0:
                return {"n": 0, "mean": None, "median": None}
            return {"n": int(len(sel)),
                    "mean": float(np.mean(sel)),
                    "median": float(np.median(sel))}

        summary_ms = {
            "variant": args.variant,
            "n_total": int(len(alive)),
            "alive_ge_1": agg(fire_alive >= 1),
        }
        for t in well_supp:
            summary_ms[f"well_supp_{t}"] = agg(fire_alive >= t)
        # sorted-rank curve for paper-Fig-5-style plotting later
        ms_sorted = sorted([float(x) for x in ms if not np.isnan(x)], reverse=True)
        summary_ms["sorted_curve"] = ms_sorted
        with open(out / "ms_summary.json", "w") as f:
            json.dump(summary_ms, f)
        ms_alive_mean = summary_ms["alive_ge_1"]["mean"]
        logger.info("[stage=ms] DONE variant=%s mean_alive=%s dt=%s",
                    args.variant,
                    f"{ms_alive_mean:.4f}" if ms_alive_mean is not None else "NaN",
                    _fmt_dur(time.time() - t0))

    # M3 — CCS  (cheap, do first)
    if "ccs" in metrics:
        logger.info("[stage=ccs] BEGIN variant=%s", args.variant)
        t0 = time.time()
        # Optional upper-bound filter: drop "always-on" latents that fire on
        # too large a fraction of N images. fire_count is per latent over all N.
        max_fire = int(round(args.ccs_max_fire_frac * N))
        ccs_alive = alive[fire[alive] <= max_fire]
        n_dropped = len(alive) - len(ccs_alive)
        logger.info("[stage=ccs] alive=%d → ccs_alive=%d (max_fire_frac=%.3f, dropped=%d)",
                    len(alive), len(ccs_alive), args.ccs_max_fire_frac, n_dropped)
        rows_ccs = compute_ccs(z, labels, ccs_alive, args.n_classes)
        _write_csv(out / "ccs_per_class.csv", rows_ccs,
                   ["class_idx", "ccs_top1", "ccs_entropy",
                    "top1_latent", "top3_latents"])
        # Latent uniqueness: # of distinct latents among the 1000 class top-1 picks.
        valid_top1 = [r["top1_latent"] for r in rows_ccs
                      if r.get("top1_latent", -1) >= 0]
        n_unique_top1 = len(set(valid_top1)) if valid_top1 else 0
        # Worst-case domination: how many classes pick the *same* top-1 latent?
        if valid_top1:
            from collections import Counter
            counts = Counter(valid_top1)
            max_class_share = counts.most_common(1)[0][1] / len(valid_top1)
            top_dominators = counts.most_common(5)
        else:
            max_class_share = 0.0
            top_dominators = []
        vals_top1 = [r["ccs_top1"] for r in rows_ccs
                     if not (isinstance(r["ccs_top1"], float) and np.isnan(r["ccs_top1"]))]
        vals_h = [r["ccs_entropy"] for r in rows_ccs
                  if not (isinstance(r["ccs_entropy"], float) and np.isnan(r["ccs_entropy"]))]
        summary_ccs = {
            "n": len(vals_top1),
            "ccs_max_fire_frac": float(args.ccs_max_fire_frac),
            "ccs_n_alive_after_filter": int(len(ccs_alive)),
            "ccs_n_dropped_by_filter": int(n_dropped),
            "ccs_top1_mean": float(np.mean(vals_top1)) if vals_top1 else None,
            "ccs_top1_median": float(np.median(vals_top1)) if vals_top1 else None,
            "ccs_entropy_mean": float(np.mean(vals_h)) if vals_h else None,
            "ccs_entropy_median": float(np.median(vals_h)) if vals_h else None,
            # Uniqueness of class top-1 latents
            "n_unique_top1": int(n_unique_top1),
            "uniqueness_ratio": float(n_unique_top1 / max(len(valid_top1), 1)),
            "max_class_share": float(max_class_share),
            "top_dominators": [(int(k), int(c)) for k, c in top_dominators],
        }
        with open(out / "ccs_summary.json", "w") as f:
            json.dump(summary_ccs, f)
        logger.info("[stage=ccs] DONE variant=%s top1_mean=%s H_mean=%s dt=%s",
                    args.variant,
                    f"{summary_ccs['ccs_top1_mean']:.4f}" if summary_ccs['ccs_top1_mean'] else "n/a",
                    f"{summary_ccs['ccs_entropy_mean']:.4f}" if summary_ccs['ccs_entropy_mean'] else "n/a",
                    _fmt_dur(time.time() - t0))

    # M2 — FMS
    if "fms" in metrics:
        logger.info("[stage=fms] BEGIN variant=%s classes=%d", args.variant, args.n_classes)
        t0 = time.time()
        rows_fms = compute_fms(
            z, labels, alive, args.n_classes, args.variant,
            args.fms_local_max_p, args.fms_neg_per_class, args.seed,
        )
        # Per-class CSV: dynamic columns covering all p in [1..local_max_p]
        p_range = list(range(1, args.fms_local_max_p + 1))
        fms_cols = ["class_idx", "acc_0"] \
                 + [f"acc_p{p}" for p in p_range] \
                 + [f"fms_local_p{p}" for p in p_range] \
                 + ["fms_global", "max_depth", "cut_features", "acc_by_depth"]
        _write_csv(out / "fms_per_class.csv", rows_fms, fms_cols)

        valid = [r for r in rows_fms
                 if not (isinstance(r["acc_0"], float) and np.isnan(r["acc_0"]))]
        if valid:
            acc_0_mean = float(np.mean([r["acc_0"] for r in valid]))
            fms_per_p = {p: float(np.mean(
                [r["acc_0"] * (r[f"fms_local_p{p}"] + r["fms_global"]) / 2
                 for r in valid])) for p in p_range}
            local_per_p = {p: float(np.mean([r[f"fms_local_p{p}"] for r in valid]))
                           for p in p_range}
            acc_p_per_p = {p: float(np.mean([r[f"acc_p{p}"] for r in valid]))
                           for p in p_range}
        else:
            acc_0_mean = float("nan")
            fms_per_p = {p: float("nan") for p in p_range}
            local_per_p = {p: float("nan") for p in p_range}
            acc_p_per_p = {p: float("nan") for p in p_range}
        summary_fms = {
            "n": len(valid),
            "acc_0_mean": acc_0_mean,
            **{f"acc_p{p}_mean": acc_p_per_p[p] for p in p_range},
            **{f"fms_local_p{p}_mean": local_per_p[p] for p in p_range},
            **{f"fms_at_{p}": fms_per_p[p] for p in p_range},
        }
        with open(out / "fms_summary.json", "w") as f:
            json.dump(summary_fms, f)
        fms_str = "  ".join(f"@{p}={fms_per_p[p]:.4f}" for p in p_range)
        logger.info("[stage=fms] DONE variant=%s %s dt=%s",
                    args.variant, fms_str, _fmt_dur(time.time() - t0))

    meta = {
        "variant": args.variant,
        "method": args.method,
        "ckpt": args.ckpt,
        "metrics": sorted(metrics),
        "well_supp_thresholds": well_supp,
        "n_samples": int(N),
        "n_alive": int(len(alive)),
        "L": fc["L"],
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("[ALL DONE] variant=%s wrote %s", args.variant, out)


if __name__ == "__main__":
    main()
