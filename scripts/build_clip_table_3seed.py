"""Build Table 1 (CC3M-trained CLIP B/32, 3-seed mean) as a tex fragment.

Reads per-seed eval JSONs from outputs/real_exp_cc3m_s{0,1,2}/<variant>/{coco,imagenet}/
and writes outputs/real_exp_cc3m_mean/table_clip_b32_3seed.tex.
"""

import argparse
import json
import statistics
from pathlib import Path


METHODS = [
    ("Shared SAE", "shared"),
    (r"+ Iso-Energy alignment loss~\citep{dhimoila2026crossmodal}", "iso_align"),
    (r"+ Group-sparse loss~\citep{kaushik2026learning}", "group_sparse"),
    ("Modality-Specific SAEs", "separated"),
    (r"+ Post-hoc Alignment (Ours)", "ours"),
]
SEEDS = [0, 1, 2]


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def load_metrics(root: Path, variant: str) -> dict:
    out = {}
    coco_recon = root / variant / "coco" / "recon.json"
    coco_ret = root / variant / "coco" / "retrieval.json"
    in_recon = root / variant / "imagenet" / "recon.json"
    in_zs = root / variant / "imagenet" / "zeroshot_raw.json"

    if coco_recon.exists():
        d = json.load(coco_recon.open())
        out["coco_mse"] = d.get("recon_error")
    if coco_ret.exists():
        d = json.load(coco_ret.open())
        i2t = d.get("I2T", {})
        t2i = d.get("T2I", {})
        for k in (1, 5, 10):
            out[f"i2t_r{k}"] = i2t.get(f"R@{k}")
            out[f"t2i_r{k}"] = t2i.get(f"R@{k}")
    if in_recon.exists():
        d = json.load(in_recon.open())
        out["in_mse"] = d.get("recon_error")
    if in_zs.exists():
        d = json.load(in_zs.open())
        out["zs_top1"] = d.get("accuracy")
    return out


def aggregate(outputs_root: Path, dataset_prefix: str = "real_exp_cc3m") -> dict:
    """Returns {variant: {metric: (mean, std)}} across SEEDS."""
    agg = {}
    for _, variant in METHODS:
        per_seed = []
        for s in SEEDS:
            r = outputs_root / f"{dataset_prefix}_s{s}"
            per_seed.append(load_metrics(r, variant))
        keys = set().union(*[m.keys() for m in per_seed])
        agg[variant] = {}
        for k in keys:
            xs = [m[k] for m in per_seed if k in m and m[k] is not None]
            if not xs:
                continue
            mu = mean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            agg[variant][k] = (mu, sd)
    return agg


def fmt_mse(x):
    return f"{x:.3f}" if x is not None else "—"


def fmt_pct(x):
    if x is None:
        return "—"
    if x <= 1.0:
        x = x * 100
    return f"{x:.2f}"


def fmt_mse_std(sd):
    return f"{sd:.4f}"


def fmt_pct_std(sd):
    return f"{sd*100:.1f}"


def render_tex(agg: dict) -> str:
    rows = []
    metric_keys = ["coco_mse", "i2t_r1", "i2t_r5", "i2t_r10",
                   "t2i_r1", "t2i_r5", "t2i_r10", "in_mse", "zs_top1"]
    minimize = {"coco_mse", "in_mse"}

    def disp_mean(k, mu):
        if mu is None:
            return None
        return fmt_mse(mu) if k.endswith("mse") else fmt_pct(mu)

    def disp_std(k, sd):
        return fmt_mse_std(sd) if k.endswith("mse") else fmt_pct_std(sd)

    def disp_key(k, mu):
        """Numeric key matching displayed precision so true ties compare equal."""
        if k.endswith("mse"):
            return float(f"{mu:.3f}")
        return float(f"{mu*100 if mu <= 1.0 else mu:.2f}")

    bold = {k: set() for k in metric_keys}
    underline = {k: set() for k in metric_keys}
    for k in metric_keys:
        keyed = []
        for _, v in METHODS:
            entry = agg[v].get(k)
            if entry is None:
                continue
            mu, _ = entry
            keyed.append((v, disp_key(k, mu)))
        if not keyed:
            continue
        cmp = (lambda x: x[1]) if k in minimize else (lambda x: -x[1])
        keyed.sort(key=cmp)
        best_val = keyed[0][1]
        bold[k] = {v for v, x in keyed if x == best_val}
        rest = [p for p in keyed if p[1] != best_val]
        if rest:
            second_val = rest[0][1]
            underline[k] = {v for v, x in rest if x == second_val}

    def cell(variant, k):
        entry = agg[variant].get(k)
        if entry is None:
            return "—"
        mu, sd = entry
        s = disp_mean(k, mu)
        if variant in bold[k]:
            s = r"\textbf{" + s + "}"
        elif variant in underline[k]:
            s = r"\underline{" + s + "}"
        return s + r" {\footnotesize $(\pm " + disp_std(k, sd) + ")$}"

    for label, variant in METHODS:
        is_ours = variant == "ours"
        prefix = r"\rowcolor{gray!20}" + "\n        " if is_ours else ""
        row = (
            f"        {prefix}{label}\n"
            f"        & {cell(variant, 'coco_mse')}\n"
            f"        & {cell(variant, 'i2t_r1')} & {cell(variant, 'i2t_r5')} & {cell(variant, 'i2t_r10')}\n"
            f"        & {cell(variant, 't2i_r1')} & {cell(variant, 't2i_r5')} & {cell(variant, 't2i_r10')}\n"
            f"        & {cell(variant, 'in_mse')} & {cell(variant, 'zs_top1')} \\\\"
        )
        rows.append(row)

    table = r"""\begin{table}[t]
    \centering
    \caption{Comparison of reconstruction and alignment quality between Post-hoc Alignment and the two auxiliary-loss baselines~\citep{dhimoila2026crossmodal, kaushik2026learning}, averaged over 3 seeds.
    Reconstruction quality is measured by mean squared error (MSE) between the input embedding and its reconstruction; cross-modal alignment is measured by Recall (R@$k$ for $k \in \{1, 5, 10\}$) for image-to-text and text-to-image retrieval on \texttt{MS-COCO}~\citep{lin2014microsoft}, and by top-1 zero-shot classification accuracy on \texttt{ImageNet1K}~\citep{deng2009imagenet}.
    Our method consistently achieves the lowest reconstruction error and the highest alignment performance.
    }
    \resizebox{0.99\textwidth}{!}{
    \begin{tabular}{l c ccc ccc c c}
        \toprule
        & \multicolumn{7}{c}{MS-COCO~\citep{lin2014microsoft}} & \multicolumn{2}{c}{ImageNet1K~\citep{deng2009imagenet}} \\
        \cmidrule(lr){2-8} \cmidrule(lr){9-10}
        & Recon. ($\downarrow$)
        & \multicolumn{3}{c}{Image-to-Text ($\uparrow$)}
        & \multicolumn{3}{c}{Text-to-Image ($\uparrow$)}
        & Recon. ($\downarrow$) & Zero-shot ($\uparrow$) \\
        \cmidrule(lr){2-2} \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-9} \cmidrule(lr){10-10}
        Methods & MSE
        & R@1 & R@5 & R@10
        & R@1 & R@5 & R@10
        & MSE & Accuracy \\
        \midrule
"""
    table += "\n".join(r + ("\n" if i < len(rows) - 1 else "") for i, r in enumerate(rows))
    table += r"""

        \bottomrule
    \end{tabular}
    }
    \label{tab:real_sae_compact}
\end{table}
"""
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default="outputs", type=Path)
    ap.add_argument("--dataset-prefix", default="real_exp_cc3m",
                    help="prefix before _s{0,1,2} (e.g. real_exp_cc3m_openclip_b)")
    ap.add_argument("--out-tex", default=None, type=Path,
                    help="output tex path (default: outputs/<prefix>_mean/table_<prefix>_3seed.tex)")
    args = ap.parse_args()

    if args.out_tex is None:
        args.out_tex = args.outputs_root / f"{args.dataset_prefix}_mean" / f"table_{args.dataset_prefix}_3seed.tex"

    agg = aggregate(args.outputs_root, dataset_prefix=args.dataset_prefix)

    # Print summary to stdout
    print(f"{'variant':<14} | {'cocoMSE':>14} | {'i2tR@1':>14} | {'t2iR@1':>14} | {'inMSE':>14} | {'zs@1':>14}")
    print("-" * 100)
    def fmt(entry, mse):
        if entry is None:
            return "—"
        mu, sd = entry
        if mse:
            return f"{mu:.3f} ±{sd:.3f}"
        return f"{mu*100:.2f}% ±{sd*100:.2f}"

    for _, v in METHODS:
        m = agg[v]
        print(
            f"{v:<14} | {fmt(m.get('coco_mse'), True):>14} | "
            f"{fmt(m.get('i2t_r1'), False):>14} | {fmt(m.get('t2i_r1'), False):>14} | "
            f"{fmt(m.get('in_mse'), True):>14} | {fmt(m.get('zs_top1'), False):>14}"
        )

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(render_tex(agg))
    print(f"\nwrote {args.out_tex}")


if __name__ == "__main__":
    main()
