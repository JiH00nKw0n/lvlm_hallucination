"""Per-atom reconstruction error: collapsed vs non-collapsed.

Uses Two SAE decoder columns as GT atom proxies.
Feeds them through Single SAE and Two SAE, compares per-atom recon error.
"""
from __future__ import annotations
import json, sys, numpy as np, torch
from pathlib import Path
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from src.models.modeling_sae import TopKSAE, TwoSidedTopKSAE

fu = sys.argv[1] if len(sys.argv) > 1 else "2"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_dir = f"outputs/real_alpha_followup_{fu}/one_sae/final"
two_dir = f"outputs/real_alpha_followup_{fu}/two_sae/final"

# Load models
one_model = TopKSAE.from_pretrained(one_dir).to(dev).eval()
two_model = TwoSidedTopKSAE.from_pretrained(two_dir).to(dev).eval()
print(f"Single SAE L={one_model.latent_size}, Two SAE Li={two_model.image_sae.latent_size}", flush=True)

# Two SAE decoder columns (GT atom proxies)
sd2 = load_file(Path(two_dir) / "model.safetensors")
Wi = sd2["image_sae.W_dec"].float()  # (L/2, d)
Wt = sd2["text_sae.W_dec"].float()   # (L/2, d)
# Single SAE decoder
sd1 = load_file(Path(one_dir) / "model.safetensors")
Ws = sd1["W_dec"].float()  # (L, d)

# Load collapse mapping
collapse = json.load(open(f"outputs/real_alpha_followup_{fu}/collapsed_pairs_comparison.json"))

# Load Hungarian matching for non-collapsed comparison
C2 = np.load(Path(two_dir) / "diagnostic_B_C_train.npy")
rates2 = np.load(Path(two_dir) / "diagnostic_B_firing_rates.npz")
from scipy.optimize import linear_sum_assignment
ai2 = np.where(rates2["rate_i"] > 0.001)[0]
at2 = np.where(rates2["rate_t"] > 0.001)[0]
r2, c2 = linear_sum_assignment(-C2[np.ix_(ai2, at2)])
oi2 = ai2[r2]; ot2 = at2[c2]
Cm2 = C2[np.ix_(ai2, at2)][r2, c2]
del C2, rates2

def atom_recon_error(atom_vec, model, modality="img"):
    """Feed a single atom vector through SAE, return ||atom - recon||^2."""
    with torch.no_grad():
        x = atom_vec.unsqueeze(0).unsqueeze(0).to(dev)  # (1, 1, d)
        if isinstance(model, TwoSidedTopKSAE):
            if modality == "img":
                out = model.image_sae(hidden_states=x)
            else:
                out = model.text_sae(hidden_states=x)
        else:
            out = model(hidden_states=x)
        recon = out.output.squeeze()  # (d,)
        err = (atom_vec.to(dev) - recon).pow(2).sum().item()
    return err

# Normalize atoms to unit norm (like the paper assumes)
Wi_n = Wi / (Wi.norm(dim=1, keepdim=True) + 1e-12)
Wt_n = Wt / (Wt.norm(dim=1, keepdim=True) + 1e-12)

print(f"\n{'='*90}")
print(f"{'':>8} | {'Two SAE img→img':>15} {'Two SAE txt→txt':>15} | {'Single←img':>12} {'Single←txt':>12} | {'Δ_img':>7} {'Δ_txt':>7}")
print(f"{'='*90}")

# === Collapsed pairs ===
print("--- COLLAPSED (merged in single SAE) ---")
col_results = []
for c in collapse:
    if c["two_pair"] is None:
        continue
    sl = c["single_lat"]
    ti, tt = c["two_pair"]
    j = c["jaccard"]

    # Two SAE atom → Two SAE recon (should be near 0)
    err_2i = atom_recon_error(Wi_n[ti], two_model, "img")
    err_2t = atom_recon_error(Wt_n[tt], two_model, "txt")
    # Two SAE atom → Single SAE recon (collapsed = bisector penalty)
    err_1i = atom_recon_error(Wi_n[ti], one_model)
    err_1t = atom_recon_error(Wt_n[tt], one_model)

    di = err_1i - err_2i
    dt = err_1t - err_2t

    col_results.append(dict(lat=sl, ti=ti, tt=tt, j=j,
                            err_2i=err_2i, err_2t=err_2t,
                            err_1i=err_1i, err_1t=err_1t,
                            di=di, dt=dt))
    print(f"  {sl:>5} | {err_2i:>15.6f} {err_2t:>15.6f} | {err_1i:>12.6f} {err_1t:>12.6f} | {di:>+7.4f} {dt:>+7.4f}  J={j:.3f}")

# === Non-collapsed pairs (C>=0.2, NOT in collapsed set) ===
print("\n--- NON-COLLAPSED (C>=0.2, Hungarian-matched) ---")
collapsed_two_img = set(c["two_pair"][0] for c in collapse if c["two_pair"] is not None)
collapsed_two_txt = set(c["two_pair"][1] for c in collapse if c["two_pair"] is not None)

noncol_results = []
cnt = 0
for pi in range(len(oi2)):
    if Cm2[pi] < 0.2:
        continue
    ti, tt = int(oi2[pi]), int(ot2[pi])
    if ti in collapsed_two_img or tt in collapsed_two_txt:
        continue

    err_2i = atom_recon_error(Wi_n[ti], two_model, "img")
    err_2t = atom_recon_error(Wt_n[tt], two_model, "txt")
    err_1i = atom_recon_error(Wi_n[ti], one_model)
    err_1t = atom_recon_error(Wt_n[tt], one_model)

    di = err_1i - err_2i
    dt = err_1t - err_2t
    noncol_results.append(dict(ti=ti, tt=tt, C=float(Cm2[pi]),
                               err_2i=err_2i, err_2t=err_2t,
                               err_1i=err_1i, err_1t=err_1t,
                               di=di, dt=dt))
    cnt += 1

print(f"  ({cnt} pairs)")

# === Summary ===
print(f"\n{'='*90}")
print(f"{'SUMMARY':>40}")
print(f"{'='*90}")

def stats(arr):
    a = np.array(arr)
    return f"mean={a.mean():.6f}  med={np.median(a):.6f}  std={a.std():.6f}"

col_di = [r["di"] for r in col_results]
col_dt = [r["dt"] for r in col_results]
nc_di = [r["di"] for r in noncol_results]
nc_dt = [r["dt"] for r in noncol_results]

print(f"\nΔ_img = err(single) - err(two)  [positive = single worse]")
print(f"  Collapsed  (n={len(col_di):>3}): {stats(col_di)}")
print(f"  Non-collap (n={len(nc_di):>3}): {stats(nc_di)}")

print(f"\nΔ_txt = err(single) - err(two)  [positive = single worse]")
print(f"  Collapsed  (n={len(col_dt):>3}): {stats(col_dt)}")
print(f"  Non-collap (n={len(nc_dt):>3}): {stats(nc_dt)}")

print(f"\nAbsolute per-atom error (single SAE):")
print(f"  Collapsed  img: {stats([r['err_1i'] for r in col_results])}")
print(f"  Collapsed  txt: {stats([r['err_1t'] for r in col_results])}")
print(f"  Non-collap img: {stats([r['err_1i'] for r in noncol_results])}")
print(f"  Non-collap txt: {stats([r['err_1t'] for r in noncol_results])}")

print(f"\nAbsolute per-atom error (two SAE):")
print(f"  Collapsed  img: {stats([r['err_2i'] for r in col_results])}")
print(f"  Collapsed  txt: {stats([r['err_2t'] for r in col_results])}")
print(f"  Non-collap img: {stats([r['err_2i'] for r in noncol_results])}")
print(f"  Non-collap txt: {stats([r['err_2t'] for r in noncol_results])}")

# Save
out_path = f"outputs/real_alpha_followup_{fu}/per_atom_recon.json"
json.dump({"collapsed": col_results, "non_collapsed": noncol_results},
          open(out_path, "w"), indent=2)
print(f"\nSaved {out_path}")
