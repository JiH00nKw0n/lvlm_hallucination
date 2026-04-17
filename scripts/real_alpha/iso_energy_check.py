"""Iso-Energy Assumption check: E[z_k^2] for image vs text per SAE latent."""
from __future__ import annotations
import json, sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap
from src.models.modeling_sae import TopKSAE
from src.datasets.cached_clip_pairs import CachedClipPairsDataset

fu = sys.argv[1] if len(sys.argv) > 1 else "2"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_dir = f"outputs/real_alpha_followup_{fu}/one_sae/final"
out_path = f"outputs/real_alpha_followup_{fu}/iso_energy.npz"

model = TopKSAE.from_pretrained(one_dir).to(dev).eval()
L = model.latent_size
print(f"SAE L={L}, device={dev}", flush=True)

ds = CachedClipPairsDataset("cache/clip_b32_coco", split="train", l2_normalize=True)
N = len(ds)
img = torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
txt = torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
print(f"N={N}, img={img.shape}, txt={txt.shape}", flush=True)

E_img = np.zeros(L, dtype=np.float64)
E_txt = np.zeros(L, dtype=np.float64)
BS = 2048
with torch.no_grad():
    for s in range(0, N, BS):
        e = min(s + BS, N)
        zi = model(hidden_states=img[s:e].unsqueeze(1).to(dev), return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        zt = model(hidden_states=txt[s:e].unsqueeze(1).to(dev), return_dense_latents=True).dense_latents.squeeze(1).cpu().numpy()
        E_img += (zi ** 2).sum(axis=0)
        E_txt += (zt ** 2).sum(axis=0)
        if (s // BS) % 50 == 0:
            print(f"  {s//BS}/{(N+BS-1)//BS}", flush=True)
E_img /= N
E_txt /= N

np.savez(out_path, E_img=E_img, E_txt=E_txt)
print(f"Saved {out_path}", flush=True)

alive = (E_img > 1e-10) | (E_txt > 1e-10)
mu = E_img[alive] / (E_img[alive] + E_txt[alive] + 1e-20)
print(f"Alive: {alive.sum()}/{L}")
print(f"mu mean={mu.mean():.4f} median={np.median(mu):.4f}")
print(f"  img-dom (mu>0.8): {(mu>0.8).sum()}")
print(f"  txt-dom (mu<0.2): {(mu<0.2).sum()}")
print(f"  bimodal (0.2~0.8): {((mu>=0.2)&(mu<=0.8)).sum()}")
ratio = np.log10(E_img[alive] / (E_txt[alive] + 1e-20))
print(f"log10(E_img/E_txt): mean={ratio.mean():.3f} std={ratio.std():.3f}")
print(f"  |log|<0.5: {(np.abs(ratio)<0.5).sum()}/{alive.sum()}")
print(f"  |log|<1.0: {(np.abs(ratio)<1.0).sum()}/{alive.sum()}")
