"""FMS evaluation on ImageNet-1K *train* (1.28M / class ≈ 1,281).

Two parallel implementations:

  --metrics tree   Paper-faithful (Härle et al. 2025, ml-research/...) tree-based
                   FMS protocol: stump in-sample acc, iterative root-feature
                   cut, depth-by-depth global. Now with N≈2600 per binary task,
                   trees no longer trivially saturate to 100%.

  --metrics probe  L1-LR (sparse linear probe) adaptation of the same FMS
                   formula: replace tree's feature_importances_ with LR
                   |coef| ranking. Same FMS aggregation. Robust to high-D
                   low-N pathology.

Per class c:
  pos = ImageNet train images of class c (~1,281)
  neg = balanced random sample from other classes (~1,281)
  X = SAE alive latent activations, y = {0,1}
  Run the chosen FMS protocol(s).

Reuses alive_idx from the val pipeline's fire_counts.json so MS/CCS/FMS
share the same alive set. Encodes SAE class-by-class to avoid materializing
the full (1.28M, L) latent matrix.

Usage:
    python scripts/real_alpha/eval_fms_train.py \\
        --variant separated --method separated \\
        --ckpt outputs/real_exp_cc3m/separated/ckpt/final \\
        --clip-cache cache/clip_b32_imagenet \\
        --val-fire-counts outputs/real_exp_cc3m/monosemanticity/separated/fire_counts.json \\
        --out outputs/real_exp_cc3m/monosemanticity/separated \\
        --metrics tree,probe \\
        --device cuda
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

warnings.filterwarnings("ignore")  # silence sklearn convergence chatter

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from tqdm import tqdm  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _fmt_dur(s: float) -> str:
    if s == float("inf"):
        return "inf"
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, required=True)
    p.add_argument("--method", choices=["shared", "separated", "aux", "ours", "vl_sae"],
                   required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--clip-cache", type=str, required=True)
    p.add_argument("--val-fire-counts", type=str, required=True,
                   help="fire_counts.json from the val pipeline (alive_idx source)")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--metrics", type=str, default="tree,probe",
                   help="Comma-separated subset of {tree,probe}")
    p.add_argument("--n-classes", type=int, default=1000)
    p.add_argument("--neg-per-class", type=int, default=1300,
                   help="Number of negatives sampled per class (balanced)")
    p.add_argument("--max-pos-per-class", type=int, default=1300,
                   help="Cap positives per class (default ~all train images of that class)")
    p.add_argument("--fms-local-max-p", type=int, default=5)
    p.add_argument("--fms-n-max", type=int, default=10,
                   help="Top-n cap for global component (paper used full max_depth)")
    p.add_argument("--lr-c", type=float, default=1.0,
                   help="Inverse regularization strength for L1 LR")
    p.add_argument("--encode-batch-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def _load_train_split(cache_dir: str) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Returns (img_ids[N], labels[N], pos_index_map). pos_index_map[img_id] = position
    in the same image_embeddings.pt order — needed because the .pt is keyed by img_id."""
    splits = json.load(open(_Path(cache_dir) / "splits.json"))
    train = splits["train"]
    img_ids = np.array([p[0] for p in train], dtype=np.int64)
    labels = np.array([p[1] for p in train], dtype=np.int64)
    return img_ids, labels, {int(iid): i for i, iid in enumerate(img_ids)}


def _load_train_embeddings(cache_dir: str, img_ids: np.ndarray) -> torch.Tensor:
    p = _Path(cache_dir) / "image_embeddings.pt"
    raw = torch.load(p, map_location="cpu", weights_only=True)
    raw = {int(k): v for k, v in raw.items()}
    out = torch.stack([raw[int(iid)] for iid in img_ids], dim=0).float()
    return out


def _pick_image_sae(model, method: str):
    if method == "shared" or method == "vl_sae":
        return model
    return model.image_sae


@torch.no_grad()
def _encode_batch(model, method: str, x_clip: torch.Tensor, device: torch.device) -> torch.Tensor:
    image_sae = _pick_image_sae(model, method)
    image_sae.eval(); image_sae.to(device)
    chunk = x_clip.to(device)
    if method == "vl_sae":
        z = image_sae.encode(chunk)
    else:
        z = image_sae(hidden_states=chunk.unsqueeze(1),
                      return_dense_latents=True).dense_latents.squeeze(1)
    return z.float().cpu()


# ----------------------------------------------------------------------
# FMS — tree (paper-faithful)
# ----------------------------------------------------------------------

def _stump_in_sample(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, int]:
    clf = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=seed)
    clf.fit(X, y)
    return float(clf.score(X, y)), int(clf.tree_.feature[0])


def fms_tree_per_class(X: np.ndarray, y: np.ndarray,
                       local_max_p: int, n_max: int, seed: int) -> dict:
    """Paper-faithful tree FMS. X has class's alive-latent activations, y in {0,1}."""
    acc_0, _ = _stump_in_sample(X, y, seed)

    # Global: max_depth=None tree, accuracy at d=1..max_d
    full = DecisionTreeClassifier(criterion="gini", random_state=seed)
    full.fit(X, y)
    max_d = int(full.tree_.max_depth)
    acc_by_depth = []
    n_use = min(n_max, max_d) if max_d > 0 else 1
    for d in range(1, n_use + 1):
        clf_d = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=seed)
        clf_d.fit(X, y)
        acc_by_depth.append(float(clf_d.score(X, y)))
    if n_use <= 1:
        fms_global = 1.0
    else:
        extra = sum(acc_by_depth[d - 1] - acc_by_depth[0] for d in range(2, n_use + 1))
        fms_global = 1.0 - extra / (n_use - 1)
    fms_global = float(np.clip(fms_global, 0.0, 1.0))

    # Local: iterative root-feature cut
    X_iter = X.copy()
    cut_features: list[int] = []
    acc_p: dict[int, float] = {}
    for p in range(1, local_max_p + 1):
        ap, root_feat = _stump_in_sample(X_iter, y, seed)
        acc_p[p] = ap
        cut_features.append(root_feat)
        X_iter[:, root_feat] = 0.0

    fms_local_p1 = float(np.clip(2.0 * (acc_0 - acc_p[1]), 0.0, 1.0))
    fms_local_p5 = float(np.clip(2.0 * (acc_0 - acc_p[5]), 0.0, 1.0)) \
                   if local_max_p >= 5 else float("nan")

    return {
        "acc_0": float(acc_0),
        "acc_p1": float(acc_p[1]),
        "acc_p5": float(acc_p.get(5, float("nan"))),
        "fms_local_p1": fms_local_p1,
        "fms_local_p5": fms_local_p5,
        "fms_global": fms_global,
        "max_depth": int(max_d),
        "cut_features": ";".join(str(int(f)) for f in cut_features),
    }


# ----------------------------------------------------------------------
# FMS — L1 sparse linear probe (paper FMS formula, LR backbone)
# ----------------------------------------------------------------------

def fms_probe_per_class(X: np.ndarray, y: np.ndarray,
                        local_max_p: int, n_max: int, lr_c: float, seed: int) -> dict:
    """L1-LR analog of FMS:
       - Top features by |coef| of full L1-LR fit.
       - acc_0 = single-feature LR (using top-1 feature), in-sample.
       - acc_p = full LR after dropping top-p features.
       - acc_cum_n = LR using top-n features only.
       Same FMS_local / FMS_global / aggregation.
    """
    def _l1(C: float) -> LogisticRegression:
        return LogisticRegression(penalty="l1", solver="liblinear", C=C,
                                  random_state=seed, max_iter=200)

    full = _l1(lr_c)
    full.fit(X, y)
    coefs = np.abs(full.coef_[0])
    order = np.argsort(-coefs)
    n_nonzero = int((coefs > 1e-9).sum())

    # acc_0: single-feature LR on top-1
    if coefs[order[0]] <= 1e-9:
        acc_0 = 0.5  # L1 zeroed everything → degenerate, no signal
    else:
        # Single-feature → effectively unregularized via large C
        clf0 = LogisticRegression(C=1e9, solver="lbfgs",
                                  random_state=seed, max_iter=200)
        clf0.fit(X[:, order[:1]], y)
        acc_0 = float(clf0.score(X[:, order[:1]], y))

    # acc_p: drop top-p, refit full L1
    acc_p: dict[int, float] = {}
    for p in (1, 5):
        if p > local_max_p:
            acc_p[p] = float("nan"); continue
        mask = np.ones(X.shape[1], dtype=bool)
        mask[order[:p]] = False
        if mask.sum() == 0:
            acc_p[p] = 0.5
        else:
            clf_r = _l1(lr_c)
            clf_r.fit(X[:, mask], y)
            acc_p[p] = float(clf_r.score(X[:, mask], y))

    # acc_cum_n: LR with top-n features only
    n_use = min(n_max, X.shape[1])
    acc_cum = []
    for n in range(1, n_use + 1):
        if n == 1:
            acc_cum.append(acc_0)
            continue
        clf_n = _l1(lr_c)
        clf_n.fit(X[:, order[:n]], y)
        acc_cum.append(float(clf_n.score(X[:, order[:n]], y)))
    if n_use <= 1:
        fms_global = 1.0
    else:
        extra = sum(acc_cum[i] - acc_0 for i in range(1, n_use))
        fms_global = 1.0 - extra / (n_use - 1)
    fms_global = float(np.clip(fms_global, 0.0, 1.0))

    fms_local_p1 = float(np.clip(2.0 * (acc_0 - acc_p[1]), 0.0, 1.0))
    fms_local_p5 = float(np.clip(2.0 * (acc_0 - acc_p[5]), 0.0, 1.0)) \
                   if not np.isnan(acc_p[5]) else float("nan")

    return {
        "acc_0": float(acc_0),
        "acc_p1": float(acc_p[1]),
        "acc_p5": float(acc_p.get(5, float("nan"))),
        "fms_local_p1": fms_local_p1,
        "fms_local_p5": fms_local_p5,
        "fms_global": fms_global,
        "n_nonzero_l1": n_nonzero,
        "top_features_by_coef": ";".join(str(int(f)) for f in order[:5]),
    }


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def _best_feature_per_row(r: dict) -> int | None:
    """Extract the row's chosen 'best latent' (= first cut feature for tree, first
    L1-coef-ranked feature for probe). Returns None if missing."""
    s = r.get("cut_features") or r.get("top_features_by_coef")
    if not s or not isinstance(s, str):
        return None
    first = s.split(";")[0]
    try:
        return int(first)
    except (ValueError, IndexError):
        return None


def _summarize_fms(rows: list[dict], label: str) -> dict:
    valid = [r for r in rows
             if not (isinstance(r.get("acc_0"), float) and np.isnan(r["acc_0"]))]
    if not valid:
        return {"protocol": label, "n": 0}
    fms1 = float(np.mean([r["acc_0"] * (r["fms_local_p1"] + r["fms_global"]) / 2
                          for r in valid]))
    fms5 = float(np.mean([r["acc_0"] * (r["fms_local_p5"] + r["fms_global"]) / 2
                          for r in valid
                          if not np.isnan(r["fms_local_p5"])]))
    # Latent uniqueness: how many distinct "best features" across the binary tasks?
    best_feats = [_best_feature_per_row(r) for r in valid]
    best_feats = [f for f in best_feats if f is not None]
    if best_feats:
        from collections import Counter
        counts = Counter(best_feats)
        n_unique = len(counts)
        max_share = counts.most_common(1)[0][1] / len(best_feats)
        top_dom = counts.most_common(5)
    else:
        n_unique = 0
        max_share = 0.0
        top_dom = []
    return {
        "protocol": label,
        "n": len(valid),
        "acc_0_mean": float(np.mean([r["acc_0"] for r in valid])),
        "acc_p1_mean": float(np.mean([r["acc_p1"] for r in valid])),
        "acc_p5_mean": float(np.mean([r["acc_p5"] for r in valid
                                       if not np.isnan(r["acc_p5"])])),
        "fms_local_p1_mean": float(np.mean([r["fms_local_p1"] for r in valid])),
        "fms_local_p5_mean": float(np.mean([r["fms_local_p5"] for r in valid
                                            if not np.isnan(r["fms_local_p5"])])),
        "fms_global_mean": float(np.mean([r["fms_global"] for r in valid])),
        "fms_at_1": fms1,
        "fms_at_5": fms5,
        "n_unique_best_feature": int(n_unique),
        "uniqueness_ratio": float(n_unique / max(len(best_feats), 1)),
        "max_class_share": float(max_share),
        "top_dominators": [(int(k), int(c)) for k, c in top_dom],
    }


def _write_csv(path: _Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = _Path(args.out); out.mkdir(parents=True, exist_ok=True)
    metrics = set(s.strip() for s in args.metrics.split(",") if s.strip())

    logger.info("[stage=load] BEGIN variant=%s ckpt=%s", args.variant, args.ckpt)
    t0 = time.time()
    model = eval_utils.load_sae(args.ckpt, args.method)

    img_ids, labels, _ = _load_train_split(args.clip_cache)
    img = _load_train_embeddings(args.clip_cache, img_ids)
    img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    logger.info("[stage=load] DONE variant=%s N_train=%d D=%d dt=%s",
                args.variant, img.shape[0], img.shape[1], _fmt_dur(time.time() - t0))

    fc = json.load(open(args.val_fire_counts))
    alive_idx = np.array(fc["alive_idx"], dtype=np.int64)
    logger.info("[stage=alive] using val alive_idx |alive|=%d", len(alive_idx))

    by_class: dict[int, np.ndarray] = {}
    for c in range(args.n_classes):
        by_class[c] = np.where(labels == c)[0]
    logger.info("Built per-class lookup: median %d images/class",
                int(np.median([len(v) for v in by_class.values()])))

    rows_tree: list[dict] = []
    rows_probe: list[dict] = []

    pbar = tqdm(range(args.n_classes), desc=f"fms_train/{args.variant}",
                mininterval=2.0, smoothing=0.1)
    cls_t0 = time.time()
    for c in pbar:
        pos_pool = by_class[c]
        if len(pos_pool) == 0:
            for L in (rows_tree, rows_probe):
                L.append({"class_idx": c, "acc_0": np.nan, "acc_p1": np.nan,
                          "acc_p5": np.nan, "fms_local_p1": np.nan,
                          "fms_local_p5": np.nan, "fms_global": np.nan})
            continue

        n_pos = min(args.max_pos_per_class, len(pos_pool))
        pos_sel = rng.choice(pos_pool, size=n_pos, replace=False)

        # Negatives: balanced random from other classes
        # build neg pool by shuffling all positions then filtering
        neg_target = min(args.neg_per_class, len(labels) - len(pos_pool))
        # Sample negatives proportional to their class size (paper-style "random other class")
        # by repeated sampling — cheap enough at 1300 size.
        neg_sel = rng.choice(np.setdiff1d(np.arange(len(labels)), pos_pool, assume_unique=False),
                              size=neg_target, replace=False)
        sel = np.concatenate([pos_sel, neg_sel])
        y = np.concatenate([np.ones(n_pos, dtype=np.int64),
                            np.zeros(neg_target, dtype=np.int64)])

        # Encode this class's batch through SAE
        x_clip = img[sel]  # CPU
        z = _encode_batch(model, args.method, x_clip, device)
        X = z[:, torch.as_tensor(alive_idx, dtype=torch.long)].numpy()

        if "tree" in metrics:
            t = fms_tree_per_class(X, y, args.fms_local_max_p, args.fms_n_max, args.seed)
            t["class_idx"] = int(c)
            rows_tree.append(t)
        if "probe" in metrics:
            p_ = fms_probe_per_class(X, y, args.fms_local_max_p, args.fms_n_max,
                                     args.lr_c, args.seed)
            p_["class_idx"] = int(c)
            rows_probe.append(p_)

        if (c + 1) % 50 == 0:
            elapsed = time.time() - cls_t0
            rate = (c + 1) / elapsed
            eta = (args.n_classes - c - 1) / rate
            logger.info("[heartbeat] class=%d/%d rate=%.2f/s eta=%s",
                        c + 1, args.n_classes, rate, _fmt_dur(eta))

    # Save per-class CSVs and summary JSONs
    summary = {
        "variant": args.variant,
        "method": args.method,
        "n_classes": args.n_classes,
        "neg_per_class": args.neg_per_class,
        "max_pos_per_class": args.max_pos_per_class,
        "n_alive": int(len(alive_idx)),
    }
    if "tree" in metrics:
        _write_csv(out / "fms_train_tree_per_class.csv", rows_tree,
                   ["class_idx", "acc_0", "acc_p1", "acc_p5",
                    "fms_local_p1", "fms_local_p5", "fms_global",
                    "max_depth", "cut_features"])
        summary["tree"] = _summarize_fms(rows_tree, "tree")
        logger.info("[summary] tree: FMS@1=%.4f FMS@5=%.4f acc_0=%.4f local@1=%.4f local@5=%.4f global=%.4f",
                    summary["tree"].get("fms_at_1", float("nan")),
                    summary["tree"].get("fms_at_5", float("nan")),
                    summary["tree"].get("acc_0_mean", float("nan")),
                    summary["tree"].get("fms_local_p1_mean", float("nan")),
                    summary["tree"].get("fms_local_p5_mean", float("nan")),
                    summary["tree"].get("fms_global_mean", float("nan")))
    if "probe" in metrics:
        _write_csv(out / "fms_train_probe_per_class.csv", rows_probe,
                   ["class_idx", "acc_0", "acc_p1", "acc_p5",
                    "fms_local_p1", "fms_local_p5", "fms_global",
                    "n_nonzero_l1", "top_features_by_coef"])
        summary["probe"] = _summarize_fms(rows_probe, "probe-l1")
        n_nz = [r["n_nonzero_l1"] for r in rows_probe if "n_nonzero_l1" in r]
        if n_nz:
            summary["probe"]["n_nonzero_l1_mean"] = float(np.mean(n_nz))
            summary["probe"]["n_nonzero_l1_median"] = float(np.median(n_nz))
        logger.info("[summary] probe: FMS@1=%.4f FMS@5=%.4f acc_0=%.4f local@1=%.4f local@5=%.4f global=%.4f l1_nz_med=%s",
                    summary["probe"].get("fms_at_1", float("nan")),
                    summary["probe"].get("fms_at_5", float("nan")),
                    summary["probe"].get("acc_0_mean", float("nan")),
                    summary["probe"].get("fms_local_p1_mean", float("nan")),
                    summary["probe"].get("fms_local_p5_mean", float("nan")),
                    summary["probe"].get("fms_global_mean", float("nan")),
                    summary["probe"].get("n_nonzero_l1_median", "n/a"))

    with open(out / "fms_train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("[ALL DONE] variant=%s wrote %s", args.variant, out)


if __name__ == "__main__":
    main()
