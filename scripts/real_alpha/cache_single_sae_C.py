"""Compute and cache single SAE's C matrix + firing rates, same as run_diagnostic_B does for two SAE."""
from __future__ import annotations
import json,sys,gc
from pathlib import Path
import numpy as np,torch
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,str(Path(__file__).resolve().parent))
import _bootstrap
from src.datasets.cached_clip_pairs import CachedClipPairsDataset
from src.models.modeling_sae import TopKSAE

fu=sys.argv[1]  # "1" or "2" or "3"
cache_dir=sys.argv[2] if len(sys.argv)>2 else "cache/clip_b32_coco"
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dir=Path(f"outputs/real_alpha_followup_{fu}/one_sae/final")

c_path=run_dir/"diagnostic_B_C_train.npy"
fr_path=run_dir/"diagnostic_B_firing_rates.npz"

if c_path.exists() and fr_path.exists():
    print(f"already cached: {c_path}, {fr_path}"); sys.exit(0)

model=TopKSAE.from_pretrained(str(run_dir)).to(dev).eval()
L=model.latent_size
ds=CachedClipPairsDataset(cache_dir,split="train",l2_normalize=True)
N=len(ds)
img=torch.stack([ds._image_dict[int(p[0])] for p in ds.pairs])
txt=torch.stack([ds._text_dict[f"{int(p[0])}_{int(p[1])}"] for p in ds.pairs])
del ds; gc.collect()
print(f"L={L}, N={N}",flush=True)

si=np.zeros(L,dtype=np.float64);st=np.zeros(L,dtype=np.float64)
sii=np.zeros(L,dtype=np.float64);stt=np.zeros(L,dtype=np.float64)
sit=np.zeros((L,L),dtype=np.float64)
fi=np.zeros(L,dtype=np.int64);ft=np.zeros(L,dtype=np.int64);cnt=0
with torch.no_grad():
    for s in range(0,N,2048):
        zi=model(hidden_states=img[s:s+2048].unsqueeze(1).to(dev),return_dense_latents=True).dense_latents.squeeze(1).cpu().double().numpy()
        zt=model(hidden_states=txt[s:s+2048].unsqueeze(1).to(dev),return_dense_latents=True).dense_latents.squeeze(1).cpu().double().numpy()
        B=zi.shape[0];si+=zi.sum(0);st+=zt.sum(0);sii+=(zi*zi).sum(0);stt+=(zt*zt).sum(0)
        sit+=zi.T@zt;fi+=(zi>0).sum(0);ft+=(zt>0).sum(0);cnt+=B
        if (s//2048)%50==0:print(f"  batch {s//2048}/{(N+2047)//2048}",flush=True)

mi=si/cnt;mt=st/cnt
C=np.nan_to_num((sit/cnt-np.outer(mi,mt))/np.sqrt(np.clip((sii/cnt-mi**2)[:,None]*(stt/cnt-mt**2)[None,:],1e-16,None)),nan=0.0)
np.save(c_path,C)
np.savez(fr_path,rate_i=fi/cnt,rate_t=ft/cnt)
print(f"saved {c_path} ({C.shape}), {fr_path}",flush=True)
