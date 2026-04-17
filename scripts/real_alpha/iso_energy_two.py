"""Iso-Energy check for Two SAE: E[z_k^2] for Hungarian-matched pairs."""
from __future__ import annotations
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from src.models.modeling_sae import TwoSidedTopKSAE
from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from scipy.optimize import linear_sum_assignment

fu = sys.argv[1] if len(sys.argv) > 1 else "2"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
two_dir = f"outputs/real_alpha_followup_{fu}/two_sae/final"
out_path = f"outputs/real_alpha_followup_{fu}/iso_energy_two.npz"

model = TwoSidedTopKSAE.from_pretrained(two_dir).to(dev).eval()
Li = model.image_sae.latent_size
Lt = model.text_sae.latent_size
print(f"Two SAE: Li={Li}, Lt={Lt}, device={dev}", flush=True)

ds = CachedClipPairsDataset("cache/clip_b32_coco", split="train", l2_normalize=True)
N = len(ds)
img = torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
txt = torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
print(f"N={N}", flush=True)

E_img = np.zeros(Li, dtype=np.float64)
E_txt = np.zeros(Lt, dtype=np.float64)
BS = 2048
with torch.no_grad():
    for s in range(0, N, BS):
        e = min(s + BS, N)
        zi = model.image_sae(hidden_states=img[s:e].unsqueeze(1).to(dev), return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        zt = model.text_sae(hidden_states=txt[s:e].unsqueeze(1).to(dev), return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        E_img += (zi ** 2).sum(axis=0)
        E_txt += (zt ** 2).sum(axis=0)
        if (s // BS) % 50 == 0:
            print(f"  {s//BS}/{(N+BS-1)//BS}", flush=True)
E_img /= N
E_txt /= N

# Hungarian matching (use cached C)
C = np.load(Path(two_dir) / "diagnostic_B_C_train.npy")
rates = np.load(Path(two_dir) / "diagnostic_B_firing_rates.npz")
ai = np.where(rates["rate_i"] > 0.001)[0]
at = np.where(rates["rate_t"] > 0.001)[0]
r, c = linear_sum_assignment(-C[np.ix_(ai, at)])
oi = ai[r]; ot = at[c]
Cm = C[np.ix_(ai, at)][r, c]

np.savez(out_path, E_img=E_img, E_txt=E_txt, oi=oi, ot=ot, Cm=Cm)
print(f"Saved {out_path}", flush=True)
print(f"Matched pairs: {len(oi)}, C>=0.2: {(Cm>=0.2).sum()}", flush=True)

# Stats for matched pairs
ei_m = E_img[oi]; et_m = E_txt[ot]
mu_m = ei_m / (ei_m + et_m + 1e-20)
print(f"\nAll matched ({len(oi)}):")
print(f"  mu: mean={mu_m.mean():.4f} median={np.median(mu_m):.4f}")

c02 = Cm >= 0.2
ei_c = E_img[oi[c02]]; et_c = E_txt[ot[c02]]
mu_c = ei_c / (ei_c + et_c + 1e-20)
print(f"C>=0.2 matched ({c02.sum()}):")
print(f"  mu: mean={mu_c.mean():.4f} median={np.median(mu_c):.4f}")
print(f"  bimodal (0.2~0.8): {((mu_c>=0.2)&(mu_c<=0.8)).sum()}/{c02.sum()}")

both = (ei_c > 1e-10) & (et_c > 1e-10)
if both.sum() > 0:
    lr = np.log10(ei_c[both] / et_c[both])
    print(f"  log10 ratio: mean={lr.mean():.3f} median={np.median(lr):.3f} std={lr.std():.3f}")
    print(f"  |log|<0.5: {(np.abs(lr)<0.5).sum()}/{both.sum()}")
    print(f"  |log|<1.0: {(np.abs(lr)<1.0).sum()}/{both.sum()}")
