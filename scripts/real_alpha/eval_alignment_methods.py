"""Compare the post-hoc alignment against the alternatives a reviewer named.

Our method links the two latent spaces with a permutation found by Hungarian
matching on co-activation. That is one choice among several, and the obvious
question is whether a different one would do better. This puts five families in
the same slot and scores them the same way.

    permutation      Hungarian on co-activation — the paper's method
                     greedy 1:1, and Hungarian on decoder cosine, as references
    transport        entropic optimal transport (Sinkhorn), which relaxes the
                     one-to-one constraint into a soft transport plan
    rotation         orthogonal Procrustes on the latent cross-covariance
    subspace         canonical correlation analysis, which drops the requirement
                     that coordinates keep their identity at all

Every operator is fit on the training split and scored on the held-out split, so
the ones with free parameters cannot buy their score by memorizing. Fitting
touches the data only through three second-moment matrices, accumulated in one
streaming pass, which keeps the whole thing in a few hundred megabytes no matter
how many pairs the training split has.

    python scripts/real_alpha/eval_alignment_methods.py \\
        --ckpt outputs/rebuttal_models/coco_k8_r1/final \\
        --dataset coco --cache-dir cache/clip_b32_coco \\
        --out outputs/rebuttal_align/coco_k8_r1
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
from scipy.optimize import linear_sum_assignment  # noqa: E402

import eval_utils  # type: ignore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", choices=["coco", "cc3m"], default="coco")
    p.add_argument("--cache-dir", default="cache/clip_b32_coco")
    p.add_argument("--eval-cache-dir", default="cache/clip_b32_coco",
                   help="where the held-out retrieval split lives")
    p.add_argument("--fit-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--max-fit-samples", type=int, default=300000)
    p.add_argument("--sinkhorn-eps", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    p.add_argument("--sinkhorn-iters", type=int, default=300)
    p.add_argument("--cca-dims", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--conf-cutoffs", type=float, nargs="+", default=[0.1, 0.2],
                   help="also score the permutation restricted to its stronger "
                        "matches, since the other operators effectively do this")
    p.add_argument("--ridge", type=float, default=1e-3,
                   help="relative ridge on the within-modality covariances for CCA")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    return p.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pair_rows(ds) -> tuple[np.ndarray, np.ndarray]:
    img = np.empty(len(ds.pairs), dtype=np.int64)
    txt = np.empty(len(ds.pairs), dtype=np.int64)
    for i, (iid, cid) in enumerate(ds.pairs):
        img[i] = ds._image_id_to_row[int(iid)]
        txt[i] = ds._text_key_to_row[ds._text_key_for(iid, cid)]
    return img, txt


@torch.no_grad()
def second_moments(model, ds, rows_i, rows_t, device, batch_size) -> dict:
    """One pass over the fitting split, keeping only what the operators need.

    Every method here is a function of the first and second moments of the two
    latent streams, so the pass accumulates those and never holds the latents
    themselves. Three L x L matrices at L=4096 is about 400 MB in float64,
    regardless of how many pairs go through.
    """
    L = int(model.image_sae.latent_size)
    acc = {
        "sum_i": torch.zeros(L, dtype=torch.float64),
        "sum_t": torch.zeros(L, dtype=torch.float64),
        "ii": torch.zeros(L, L, dtype=torch.float64),
        "tt": torch.zeros(L, L, dtype=torch.float64),
        "it": torch.zeros(L, L, dtype=torch.float64),
        "fire_i": torch.zeros(L, dtype=torch.float64),
        "fire_t": torch.zeros(L, dtype=torch.float64),
    }
    model.eval().to(device)
    n = len(rows_i)
    for s in range(0, n, batch_size):
        xi = ds._image_table[torch.as_tensor(rows_i[s:s + batch_size])].to(device)
        xt = ds._text_table[torch.as_tensor(rows_t[s:s + batch_size])].to(device)
        zi = model.image_sae(hidden_states=xi.unsqueeze(1),
                             return_dense_latents=True).dense_latents.squeeze(1).float()
        zt = model.text_sae(hidden_states=xt.unsqueeze(1),
                            return_dense_latents=True).dense_latents.squeeze(1).float()
        acc["sum_i"] += zi.sum(0).cpu().double()
        acc["sum_t"] += zt.sum(0).cpu().double()
        acc["ii"] += (zi.T @ zi).cpu().double()
        acc["tt"] += (zt.T @ zt).cpu().double()
        acc["it"] += (zi.T @ zt).cpu().double()
        acc["fire_i"] += (zi != 0).sum(0).cpu().double()
        acc["fire_t"] += (zt != 0).sum(0).cpu().double()
        if (s // batch_size) % 20 == 0:
            logger.info("  moments %d/%d", s, n)
    acc["n"] = float(n)
    return acc


def centered(acc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered covariances Sii, Stt, Sit."""
    n = acc["n"]
    mi = (acc["sum_i"] / n).numpy()
    mt = (acc["sum_t"] / n).numpy()
    Sii = acc["ii"].numpy() / n - np.outer(mi, mi)
    Stt = acc["tt"].numpy() / n - np.outer(mt, mt)
    Sit = acc["it"].numpy() / n - np.outer(mi, mt)
    return Sii, Stt, Sit


def correlation_from(Sii, Stt, Sit) -> np.ndarray:
    di = np.sqrt(np.clip(np.diag(Sii), 1e-12, None))
    dt = np.sqrt(np.clip(np.diag(Stt), 1e-12, None))
    return Sit / np.outer(di, dt)


def hungarian_perm(C, alive_i, alive_t) -> np.ndarray:
    Cm = np.array(C, dtype=np.float64, copy=True)
    Cm[~alive_i, :] = -1e9
    Cm[:, ~alive_t] = -1e9
    Cm = np.nan_to_num(Cm, nan=-1e9, posinf=1.0, neginf=-1e9)
    row, col = linear_sum_assignment(-Cm)
    perm = np.zeros(C.shape[0], dtype=np.int64)
    perm[row] = col
    return perm


def greedy_perm(C, alive_i, alive_t) -> np.ndarray:
    """Take the best remaining pair repeatedly. A weaker matcher on purpose."""
    ri, rt = np.where(alive_i)[0], np.where(alive_t)[0]
    sub = np.nan_to_num(C[np.ix_(ri, rt)], nan=-1e9)
    perm = np.zeros(C.shape[0], dtype=np.int64)
    order = np.dstack(np.unravel_index(np.argsort(-sub, axis=None), sub.shape))[0]
    used_r, used_c = set(), set()
    for a, b in order:
        if a in used_r or b in used_c:
            continue
        used_r.add(a)
        used_c.add(b)
        perm[ri[a]] = rt[b]
        if len(used_r) == min(len(ri), len(rt)):
            break
    return perm


def dec_cos_perm(model, alive_i, alive_t) -> np.ndarray:
    Wi = model.image_sae.W_dec.detach().cpu().float().numpy()
    Wt = model.text_sae.W_dec.detach().cpu().float().numpy()
    Wi = Wi / (np.linalg.norm(Wi, axis=1, keepdims=True) + 1e-12)
    Wt = Wt / (np.linalg.norm(Wt, axis=1, keepdims=True) + 1e-12)
    return hungarian_perm(Wi @ Wt.T, alive_i, alive_t)


def sinkhorn_plan(C, alive_i, alive_t, eps, iters, device) -> np.ndarray:
    """Entropic optimal transport with uniform marginals on the alive block."""
    ri, rt = np.where(alive_i)[0], np.where(alive_t)[0]
    K = torch.as_tensor(np.nan_to_num(C[np.ix_(ri, rt)], nan=0.0),
                        dtype=torch.float32, device=device) / eps
    n, m = K.shape
    f = torch.zeros(n, device=device)
    g = torch.zeros(m, device=device)
    log_a = -np.log(n)
    log_b = -np.log(m)
    for _ in range(iters):
        f = log_a - torch.logsumexp(K + g[None, :], dim=1)
        g = log_b - torch.logsumexp(K + f[:, None], dim=0)
    T_sub = torch.exp(K + f[:, None] + g[None, :]).cpu().numpy()
    T = np.zeros(C.shape, dtype=np.float32)
    T[np.ix_(ri, rt)] = T_sub
    return T


def procrustes_map(Sit: np.ndarray) -> np.ndarray:
    """Orthogonal matrix carrying the text latent space onto the image one.

    Maximizes trace(R^T Sit^T) over orthogonal R, which is the same objective
    the permutation solves, with the one-to-one constraint replaced by
    orthogonality. It is therefore the natural relaxation to compare against.
    """
    u, _s, vt = np.linalg.svd(Sit, full_matrices=False)
    return (u @ vt).T          # applied as z_t @ R


def cca_maps(Sii, Stt, Sit, dim, ridge) -> tuple[np.ndarray, np.ndarray]:
    """Canonical directions for the two latent spaces.

    Unlike the other methods this does not keep coordinates identifiable: it
    finds a shared subspace and both sides are projected into it. That is a
    weaker claim about interpretability, which is exactly why it is worth
    checking whether it retrieves better.
    """
    ti = ridge * np.trace(Sii) / Sii.shape[0]
    tt = ridge * np.trace(Stt) / Stt.shape[0]
    Ai = Sii + ti * np.eye(Sii.shape[0])
    At = Stt + tt * np.eye(Stt.shape[0])
    Li = np.linalg.cholesky(Ai)
    Lt = np.linalg.cholesky(At)
    M = np.linalg.solve(Li, Sit)
    M = np.linalg.solve(Lt, M.T).T
    u, _s, vt = np.linalg.svd(M, full_matrices=False)
    A = np.linalg.solve(Li.T, u[:, :dim])
    B = np.linalg.solve(Lt.T, vt[:dim].T)
    return A, B


@torch.no_grad()
def encode_split(model, ds, rows, side, device, batch_size) -> torch.Tensor:
    sae = model.image_sae if side == "image" else model.text_sae
    table = ds._image_table if side == "image" else ds._text_table
    out = []
    for s in range(0, len(rows), batch_size):
        x = table[torch.as_tensor(rows[s:s + batch_size])].to(device)
        z = sae(hidden_states=x.unsqueeze(1), return_dense_latents=True).dense_latents.squeeze(1)
        out.append(z.float().cpu())
    return torch.cat(out)


def recall(z_img, z_txt, pair_img_idx, gt_caps) -> dict:
    zi = z_img / z_img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    zt = z_txt / z_txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    t2i = np.empty(zt.shape[0], dtype=np.int64)
    for s in range(0, zt.shape[0], 1024):
        sc = zt[s:s + 1024] @ zi.T
        gt = pair_img_idx[s:s + 1024]
        gts = sc[np.arange(len(gt)), gt]
        t2i[s:s + 1024] = ((sc >= gts[:, None]).sum(dim=1) - 1).cpu().numpy()
    i2t = np.empty(zi.shape[0], dtype=np.int64)
    for s in range(0, zi.shape[0], 512):
        sc = zi[s:s + 512] @ zt.T
        for r in range(sc.shape[0]):
            i = s + r
            best = sc[r, gt_caps[i]].max()
            i2t[i] = int((sc[r] >= best).sum().item()) - 1
    out = {}
    for k in (1, 5, 10):
        out[f"I2T R@{k}"] = float((i2t < k).mean())
        out[f"T2I R@{k}"] = float((t2i < k).mean())
    return out


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = eval_utils.load_sae(args.ckpt, "separated")

    # ---- fit ---------------------------------------------------------------
    ds_fit = eval_utils.load_pair_dataset(args.cache_dir, args.dataset, split=args.fit_split)
    ri, rt = pair_rows(ds_fit)
    if args.max_fit_samples and args.max_fit_samples < len(ri):
        sel = np.linspace(0, len(ri) - 1, args.max_fit_samples, dtype=np.int64)
        ri, rt = ri[sel], rt[sel]
    logger.info("fitting on %d pairs from %s", len(ri), args.fit_split)
    acc = second_moments(model, ds_fit, ri, rt, device, args.batch_size)
    Sii, Stt, Sit = centered(acc)
    C = correlation_from(Sii, Stt, Sit)
    alive_i = acc["fire_i"].numpy() > 0
    alive_t = acc["fire_t"].numpy() > 0
    logger.info("alive image %d, text %d", int(alive_i.sum()), int(alive_t.sum()))

    # ---- held-out split ----------------------------------------------------
    ds_ev = eval_utils.load_pair_dataset(args.eval_cache_dir, "coco", split=args.eval_split)
    pairs = ds_ev.pairs
    uniq = sorted({int(p[0]) for p in pairs})
    pos = {iid: i for i, iid in enumerate(uniq)}
    pair_img_idx = np.array([pos[int(p[0])] for p in pairs], dtype=np.int64)
    gt_caps: list[list[int]] = [[] for _ in uniq]
    for ci, gi in enumerate(pair_img_idx):
        gt_caps[int(gi)].append(ci)
    img_rows = np.array([ds_ev._image_id_to_row[i] for i in uniq], dtype=np.int64)
    txt_rows = np.array([ds_ev._text_key_to_row[ds_ev._text_key_for(int(p[0]), int(p[1]))]
                         for p in pairs], dtype=np.int64)
    z_img = encode_split(model, ds_ev, img_rows, "image", device, args.batch_size)
    z_txt = encode_split(model, ds_ev, txt_rows, "text", device, args.batch_size)
    logger.info("held-out: %d images, %d captions", z_img.shape[0], z_txt.shape[0])

    results: dict[str, dict] = {}

    def score(name: str, zi: torch.Tensor, zt: torch.Tensor) -> None:
        results[name] = recall(zi, zt, pair_img_idx, gt_caps)
        r = results[name]
        logger.info("%-22s I2T R@1 %.4f  T2I R@1 %.4f", name, r["I2T R@1"], r["T2I R@1"])

    perm = hungarian_perm(C, alive_i, alive_t)
    score("hungarian (ours)", z_img, z_txt[:, torch.as_tensor(perm)])

    # The other operators all shrink the working set: CCA keeps a few hundred
    # directions, Sinkhorn with a non-zero epsilon spreads mass and damps weak
    # coordinates. The permutation keeps every coordinate, including the ones
    # whose match is near noise, so this arm gives it the same freedom.
    matched_c = np.where(alive_i, C[np.arange(len(perm)), perm], -np.inf)
    for cmin in args.conf_cutoffs:
        keep = torch.as_tensor(matched_c >= cmin, dtype=torch.bool)
        n_keep = int(keep.sum())
        if n_keep < 8:
            continue
        score(f"hungarian, matches c>={cmin} ({n_keep} coords)",
              z_img[:, keep], z_txt[:, torch.as_tensor(perm)][:, keep])
    score("greedy 1:1", z_img, z_txt[:, torch.as_tensor(greedy_perm(C, alive_i, alive_t))])
    score("hungarian on decoder cosine", z_img,
          z_txt[:, torch.as_tensor(dec_cos_perm(model, alive_i, alive_t))])

    for eps in args.sinkhorn_eps:
        T = sinkhorn_plan(C, alive_i, alive_t, eps, args.sinkhorn_iters, device)
        Tn = T / np.clip(T.sum(axis=1, keepdims=True), 1e-12, None)
        score(f"sinkhorn eps={eps}", z_img, z_txt @ torch.as_tensor(Tn.T, dtype=torch.float32))

    R = procrustes_map(Sit)
    score("procrustes (rotation)", z_img, z_txt @ torch.as_tensor(R, dtype=torch.float32))

    for d in args.cca_dims:
        A, B = cca_maps(Sii, Stt, Sit, d, args.ridge)
        score(f"CCA d={d}",
              z_img @ torch.as_tensor(A, dtype=torch.float32),
              z_txt @ torch.as_tensor(B, dtype=torch.float32))

    report = {
        "ckpt": args.ckpt, "dataset": args.dataset,
        "fit_split": args.fit_split, "n_fit_pairs": int(len(ri)),
        "eval_split": args.eval_split,
        "n_alive_image": int(alive_i.sum()), "n_alive_text": int(alive_t.sum()),
        "results": results,
    }
    (out / "alignment_methods.json").write_text(json.dumps(report, indent=2))

    print()
    hdr = f"{'method':<28}{'I2T R@1':>9}{'I2T R@5':>9}{'T2I R@1':>9}{'T2I R@5':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<28}{100 * r['I2T R@1']:>9.2f}{100 * r['I2T R@5']:>9.2f}"
              f"{100 * r['T2I R@1']:>9.2f}{100 * r['T2I R@5']:>9.2f}")
    print(f"\nwrote {out / 'alignment_methods.json'}")


if __name__ == "__main__":
    main()
